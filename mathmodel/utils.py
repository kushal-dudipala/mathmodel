import os

SCRATCH_DIR = "/home/hice1/ychauhan9/scratch"

def get_scratch_dir():
    return SCRATCH_DIR

def get_huggingface_token():
    return os.getenv("HUGGINGFACE_TOKEN")