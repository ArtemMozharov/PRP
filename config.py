import json
import os

class Config:
    """config class for experiments"""
    def __init__(self, experiments, experiments_path):
        self.experiments = experiments
        self.experiments_path = experiments_path
class Experiment:
    """experiment class"""
    def __init__(
        self,
        model,
        load_model_dir,
        load_model_name,
        test_image_dir,
        write_path,
        base_architecture,
        img_size,
        prototype_shape,
        num_classes,
        prototype_activation_function,
        add_on_layers_type
    ):
        self.model=model
        self.load_model_dir=load_model_dir
        self.load_model_name=load_model_name
        self.test_image_dir=test_image_dir
        self.write_path=write_path
        self.base_architecture=base_architecture
        self.img_size=img_size
        self.num_classes=num_classes
        self.prototype_shape=(prototype_shape[0], prototype_shape[1], prototype_shape[2], prototype_shape[3] )
        self.prototype_activation_function=prototype_activation_function
        self.add_on_layers_type=add_on_layers_type



def load_config(config_path, experiments_path):
    if not config_path:
        config_path="config.json"

    with open(config_path, "r") as file:
        data = json.load(file)
        experiments = [Experiment(**experiment) for experiment in data]
        return Config(experiments, experiments_path)

