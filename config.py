from pathlib import Path

def get_config():
    return{
        "batch_size" : 3,
        "num_epochs" : 15,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang-src": "en",
        "lang-tgt": "it",
        "model_folder" : "weights",
        "model_filename" : "tmodel",
        "model_basename" : "tmodel_",
        "tokenizer_file" : "tokenizer_{0}.json",
        "experiment_name" : "runs/tmodel",
        "preload": None
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config[('model_folder')]
    model_basename = config[('model_basename')]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)