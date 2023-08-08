#!/usr/bin/env python3

from os import listdir, path, remove
from sys import argv 
from pathlib import Path

files = listdir(argv[1])

xml_files = [f for f in files if Path(f).suffix == '.xml']
xml_file_stems = [Path(f).with_suffix('') for f in xml_files]

files_to_remove = [f for f in files if Path(f).with_suffix('') not in xml_file_stems] 

for f in files_to_remove:
    path_to_remove = Path(argv[1]) / f
    if path.isfile(path_to_remove.with_suffix('.xml')):
        print(f"Found {path_to_remove.with_suffix('.xml')}")
    else:
        print(f"Removing {path_to_remove}")
        remove(path_to_remove)