import json

class Config:
    """config class for experiments""""
    def __init__(self, experiments):
        experiments = experiments

def load_config(config_path):
    if not config_path:
        config_path="config.json"
    with open(config_path, "r") as file:
        experiments = json.load(file)
        return Config(experiments)
    

