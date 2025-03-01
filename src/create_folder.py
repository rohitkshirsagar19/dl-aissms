import os
import shutil
import argparse
import yaml
import pandas as pd
import numpy as np
from get_data import get_data , read_params

def create_fold(config , image=None):
    config=get_data(config)
    dirr = config['load_data']['preprocessed_data']
    cla = config['load_data']['num_classes']
    
    # Check if 'train' and 'test' directories already exist
    if os.path.exists(dirr + '/train') and os.path.exists(dirr + '/test'):
        print("Train and Test folder already exists")
        print("Skipping folder creation...")
    else:
        # Create 'train' and 'test' folders
        os.makedirs(dirr + '/train', exist_ok=True)
        os.makedirs(dirr + '/test', exist_ok=True)
        
        # Create subdirectories for each class
        for i in range(cla):
            os.makedirs(f"{dirr}/train/class_{i}", exist_ok=True)
            os.makedirs(f"{dirr}/test/class_{i}", exist_ok=True)

        print("Folders have been created.")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    pased_args=args.parse_args()
    create_fold(config=pased_args.config)