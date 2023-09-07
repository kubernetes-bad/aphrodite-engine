import time
import copy
from functools import partial
from typing import Any, List, Optional, TYPE_CHECKING, Tuple

from aphrodite.common.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig
from aphrodite.processing.scheduler import Scheduler
from aphrodite.engine.args_tools import EngineArgs
from aphrodite.engine.ray_tools import initialize_cluster, ray, RayWorker
from aphrodite.common.logger import init_logger
from aphrodite.common.outputs import RequestOutput
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.common.sequence import Sequence, SequenceGroup, SequenceStatus
from aphrodite.transformers_utils.tokenizer import detokenize_incrementally, get_tokenizer
from aphrodite.common.utils import Counter

if ray:
    from ray.air.util.torch_dist import init_torch_dist_process_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup


logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 5 

class AphroditeEngine:
    """An engine that receives requests and generates text!

    This is the main class for the Aphrodite Engine. It receives requests from clients
    and generates texts from the model. It includes a tokenizer, a language model
    (possibly distributed across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). This class utilizes iteration-level scheduling and efficient memory
    management to maximize the serving throughput.

    NOTE: The config arguments are derived from the `EngineArgs` class. For the comprehensive list
    of arguments, see `EngineArgs`.

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        distributed_init_method: The initialization method for distributed
            execution. See `torch.distributed.init_process_group` for details.
        stage_devices: The list of devices for each stage. Each stage is a list
            of (rank, node_resource, device) tuples.
        log_stats: Whether to log statistics.    
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        distributed_init_method: str,
        placement_group: Optional["PlacementGroup"],
        log_stats: bool,
    ) -> None:
        logger.info(
            "Initializing Aphrodite Engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"use_dummy_weights={model_config.use_dummy_weights}, "
            f"download_dir={model_config.download_dir!r}, "
            f"use_np_weights={model_config.use_np_weights}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"seed={model_config.seed})")

        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats
        self._verify_args()

        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code)
        self.seq_counter = Counter()

        if self.parallel_config.worker_use_ray:
            self._init_workers_ray(placement_group)
        else:
            self._init_workers(distributed_init_method)

        self._init_cache()

        self.scheduler = Scheduler(scheduler_config, cache_config)

        self.last_logging_time = 0.0
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        self.num_generation_tokens: List[Tuple[float, int]] = []

    def _init_workers(self, distributed_init_method: str):
        from aphrodite.task_handler.worker import Worker

        assert self.parallel_config.world_size == 1, (
            "Ray is required if parallel_config.world_size > 1.")
        
        self.workers : List[Worker] = []
        worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            0,
            distributed_init_method,
        )
        self.workers.append(worker)
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )

    def _init_workers_ray(self, placement_group: "PlacementGroup"):
        from aphrodite.task_handler.worker import Worker

        self.workers: List[Worker] = []
        for bundle in placement_group.bundle_specs:
            if not bundle.get("GPU", 0):
                continue
            worker = ray.remote(
                num_cpus=0,
                num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True),
            )(RayWorker).remote()
            self.workers.append(worker)
        
        init_torch_dist_process_group(self.workers, backend="nccl")
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        self._run_workers("init_worker",
                          get_all_outputs=True,
                          worker_init_fn=lambda: Worker(
                                model_config,
                                parallel_config,
                                scheduler_config,
                                None,
                                None,
                          ))
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )
        
    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    def _init_cache(self) -> None:
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)
        # TODO: Change to debug log.
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")
        
        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing the `gpu_memory_utilization` "
                             "when initializing Aphrodite.")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._run_workers("init_cache_engine", cache_config=self.cache_config)

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "AphroditeEngine":
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        distributed_init_method, placement_group = initialize_cluster(parallel_config)
        # Create the engine.
        engine = cls(*engine_configs,
                     distributed_init_method,
                     placement_group,
                     log_stats=not engine_args.disable_log_stats)
        return engine

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current time.
        """
        if arrival_time is None:
            arrival_time = time.time()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)

        block_size = self.cache_config.block_size
        seqs: List[Sequence] = []
        for _ in range(sampling_params.best_of):
            seq_id = next(self.seq_counter)
            seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)
            seqs.append(seq)

        seq_group = SequenceGroup(request_id, seqs, sampling_params, arrival_time)

        self.scheduler.add_seq_group(seq_group)

    def abort_request(self, request_id: str) -> None:
        """Aborts a request with the given ID.
        Args:
            request_id: The ID of the request to abort.
        """
        self.scheduler.abort_seq_group(request_id)

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests"""
        return self.scheduler.has_unfinished_seqs()

    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be seapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        # (seq_group_metadata_list, scheduler_outputs,
        #  ignored_seq_groups) = self.scheduler.schedule()
        # if ((not seq_group_metadata_list) and scheduler_outputs.is_empty()
        #         and (not ignored_seq_groups)):
        #     # Nothing to do.
        #     return []

        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        if scheduler_outputs.is_empty():
            if not scheduler_outputs.ignored_seq_groups:
                return []
            return [
                RequestOutput.from_seq_group(seq_group)
                for seq_group in scheduler_outputs.ignored_seq_groups
            ]


        output = self._run_workers(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )
        seq_groups = self.scheduler.update(output)

        self._decode_sequences(seq_groups)              # Decode the sequence
        self._stop_sequences(seq_groups)                # Stop the sequences that meet the stopping criteria
        self.scheduler.free_finished_seq_groups()       # Free the finished sequence groups.

        request_outputs: List[RequestOutput] = []
        for seq_group in seq_groups + scheduler_outputs.ignored_seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        if self.log_stats:
            self._log_system_stats(scheduler_outputs.prompt_run,
                                   scheduler_outputs.num_batched_tokens)
        return request_outputs
    
    def _log_system_stats(
            self,
            prompt_run: bool,
            num_batched_tokens: int,
    ) -> None:
        now = time.time()
        if prompt_run:
            self.num_prompt_tokens.append((now, num_batched_tokens))
        else:
            self.num_generation_tokens.append((now, num_batched_tokens))

        elapsed_time = now - self.last_logging_time
        if elapsed_time < _LOGGING_INTERVAL_SEC:
            return
        
        self.num_prompt_tokens = [(t, n) for t, n in self.num_prompt_tokens
                                  if now - t < _LOGGING_INTERVAL_SEC]
        self.num_generation_tokens = [(t, n)
                                      for t, n in self.num_generation_tokens
                                      if now - t < _LOGGING_INTERVAL_SEC]
        
        if len(self.num_prompt_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_prompt_tokens[:-1])
            window = now - self.num_prompt_tokens[0][0]
            avg_prompt_throughput = total_num_tokens / window
        else:
            avg_prompt_throughput = 0.0
        if len(self.num_generation_tokens) > 1:
            total_num_tokens = sum(n
                                   for _, n in self.num_generation_tokens[:-1])
            window = now - self.num_generation_tokens[0][0]
            avg_generation_throughput = total_num_tokens / window
        else:
            avg_generation_throughput = 0.0

        total_num_gpu_blocks = self.cache_config.num_gpu_blocks
        num_free_gpu_blocks = (
            self.scheduler.block_manager.get_num_free_gpu_blocks())
        num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
        gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks

        total_num_cpu_blocks = self.cache_config.num_cpu_blocks
        if total_num_cpu_blocks > 0:
            num_free_cpu_blocks = (
                self.scheduler.block_manager.get_num_free_cpu_blocks())
            num_used_cpu_blocks = total_num_cpu_blocks = num_free_cpu_blocks
            cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
        else:
            cpu_cache_usage = 0.0

        logger.info("Avg prompt throughput: "
                    f"{avg_prompt_throughput:.1f} tokens/s, "
                    "Avg generation throughput: "
                    f"{avg_generation_throughput:.1f} tokens/s, "
                    f"Running: {len(self.scheduler.running)} reqs, "
                    f"Swapped: {len(self.scheduler.swapped)} reqs, "
                    f"Pending {len(self.scheduler.waiting)} reqs, "
                    f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}%, "
                    f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%")
        self.last_logging_time = now

    def _decode_sequences(self, seq_groups: List[SequenceGroup]) -> None:
        for seq_group in seq_groups:
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                last_token_id = seq.get_last_token_id()
                if last_token_id is not None:
                    new_token, new_output_text = detokenize_incrementally(
                        self.tokenizer,
                        seq.output_tokens,
                        last_token_id,
                        skip_special_tokens=True,
                    )
                    if new_token is not None:
                        seq.output_tokens.append(new_token)
                        seq.output_text = new_output_text


    def _stop_sequences(self, seq_groups: List[SequenceGroup]) -> None:
        """Stop the finished sequences."""
        for seq_group in seq_groups:
            sampling_params = seq_group.sampling_params
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                stopped = False
                for stop_str in sampling_params.stop:
                    if seq.output_text.endswith(stop_str):
                        seq.output_text = seq.output_text[:-len(stop_str)]              # Truncate the output text so that the stop string isn't included in the output.
                        self.scheduler.free_seq(seq, SequenceStatus.FINISHED_STOPPED)
                        stopped = True
                        break
                if stopped:
                    continue
                
                if (seq.get_len() >=
                        self.scheduler_config.max_model_len):
                    self.scheduler.free_seq(
                        seq, SequenceStatus.FINISHED_LENGTH_CAPPED)
                    continue
                if seq.get_output_len() == sampling_params.max_tokens:
                    self.scheduler.free_seq(
                        seq, SequenceStatus.FINISHED_LENGTH_CAPPED)
                    continue
                if not sampling_params.ignore_eos:
                    if seq.get_last_token_id() == self.tokenizer.eos_token_id:
                        self.scheduler.free_seq(
                            seq, SequenceStatus.FINISHED_STOPPED)
                        continue

    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            # executor = getattr(worker, method)
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, method)    # FIXME: will this break us?
            else:
                executor = getattr(worker, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)

        if get_all_outputs:
            return all_outputs

        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output
                
    
    