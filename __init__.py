from .prompt_manager import PromptManager
from .router_data import (
    prepare_support_query_split,
    create_router_training_data,
    save_few_shot_memory,
)


def __getattr__(name):
    if name == "AffinityProbe":
        from .affinity_probe import AffinityProbe
        return AffinityProbe
    if name == "EWCRegularizer":
        from .ewc import EWCRegularizer
        return EWCRegularizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
