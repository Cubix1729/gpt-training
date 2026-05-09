"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np

from huggingface_hub import snapshot_download

local_dir = "edu_fineweb10B"
dataset_url = "sample-10BT"

# create the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the pretokenized shards
snapshot_download(
    repo_id="ShallowU/FineWeb-Edu-10B-Tokens-NPY",
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns="*.npy"
)
