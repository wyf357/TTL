from openttl.data.adapt_eval import build_train_dataset, load_raw_dataset
from openttl.data.mmstar import load_mmstar_dataset, load_mmstar_table
from openttl.data.stream import batched_stream, iter_hf_dataset

# ERQA 依赖 tensorflow；BFCL 等文本评测不应被该依赖挡住。
try:
    from openttl.data.erqa import load_erqa_dataset, get_erqa_dataset_size, parse_erqa_example
except ImportError:  # pragma: no cover
    load_erqa_dataset = None
    get_erqa_dataset_size = None
    parse_erqa_example = None
