#!/usr/bin/env python3
# credit where it's due: GitHub Copilot (completion: wrote this for me)videos
TRAIN_PERCENTAGE = 0.8
VALID_PERCENTAGE = 0.1
TEST_PERCENTAGE = 0.1

import os
import shutil
import random

def main():
    # Create directories
    os.makedirs('train', exist_ok=True)
    os.makedirs('valid', exist_ok=True)
    os.makedirs('test', exist_ok=True)

    # Get all the files
    all_files = os.listdir('videos')
    random.shuffle(all_files)

    # Split into train, valid, and test
    train_cutoff = int(len(all_files) * TRAIN_PERCENTAGE)
    valid_cutoff = int(len(all_files) * (TRAIN_PERCENTAGE + VALID_PERCENTAGE))

    train_files = all_files[:train_cutoff]
    valid_files = all_files[train_cutoff:valid_cutoff]
    test_files = all_files[valid_cutoff:]

    # Move files to correct directories
    for filename in train_files:
        shutil.move(os.path.join('videos', filename), os.path.join('train', filename))

    for filename in valid_files:
        shutil.move(os.path.join('videos', filename), os.path.join('valid', filename))

    for filename in test_files:
        shutil.move(os.path.join('videos', filename), os.path.join('test', filename))

if __name__ == '__main__':
    main()