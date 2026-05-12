"""
Pre-download all HuggingFace datasets and the BERT tokenizer used by the text
sweep experiments.  Run this once (on a login node or in a setup job) before
submitting slurm_text_sweep.sh so that the array tasks do not race on the
shared cache directory.

Usage:
    python download_text_datasets.py
"""

from datasets import load_dataset
from transformers import AutoTokenizer

DATASETS = [
    ('glue',         'rte'),
    ('glue',         'mrpc'),
    ('glue',         'cola'),
    ('glue',         'sst2'),
    ('google/boolq', None),
    ('ag_news',      None),
]

TOKENIZER = 'bert-base-uncased'


def main() -> None:
    print(f"Downloading tokenizer: {TOKENIZER}")
    AutoTokenizer.from_pretrained(TOKENIZER)
    print(f"  done.\n")

    for hf_path, hf_config in DATASETS:
        label = hf_config or hf_path
        print(f"Downloading {label} ...")
        kwargs = dict(split='train', trust_remote_code=False)
        if hf_config:
            load_dataset(hf_path, hf_config, **kwargs)
        else:
            load_dataset(hf_path, **kwargs)
        print(f"  done.\n")

    print("All datasets cached successfully.")


if __name__ == '__main__':
    main()
