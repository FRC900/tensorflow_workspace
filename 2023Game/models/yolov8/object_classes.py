# Load object classes from config.yaml file,
# store them in a way which can be indexed by class number
import yaml

class ObjectClasses :
    def __init__(self, config_path):
        with open(config_path, 'r') as yaml_file:
            yaml_map = yaml.safe_load(yaml_file)
            max_key = max(yaml_map['names'].keys())
            self._class_names = [yaml_map['names'][i] for i in range(max_key + 1)]

    def get_name(self, index):
        return self._class_names[index]

    def __len__(self):
        return len(self._class_names)