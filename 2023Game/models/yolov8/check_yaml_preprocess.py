# YOLO embeds data prep script in a yaml config file
# Use this to run that script standalone

import yaml as y

with open("FRC2023.yaml", 'r') as file:
    yaml = y.safe_load(file)

exec(yaml['download'])
