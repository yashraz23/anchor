"""Scheduler configuration."""


class SchedulerConfig:
    """Configuration for the vLLM scheduler."""

    max_num_seqs: int = 128
    """Maximum number of sequences to be processed in a single iteration."""

    enable_chunked_prefill: bool = True
    """Whether to split a large prefill into smaller chunks."""


class EngineArgs:
    """Engine arguments exposed on the command line."""

    @staticmethod
    def add_cli_args(parser):
        group = parser.add_argument_group("engine")
        group.add_argument("--max-num-seqs", type=int, default=None)
        group.add_argument("--enforce-eager", action="store_true")
        group.add_argument("--gpu-memory-utilization", type=float, default=0.9)
