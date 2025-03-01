import os
import shutil
import argparse
import yaml
import pandas as pd
import numpy as np
from get_data import get_data, read_params

################# CREATING FOLDER - START ################
def create_fold(config, image=None):
    config=get_data(config)
    dirr = config['load_data']['preproseesd_data']
    cla = config['load_data']['num_classes']
    #print(dirr)
    #print(cla)
    if os.path.exists(dirr+'/'+'train'+'class_0') and os.path.exists(dirr+'/'+'test'+'class_0'):
        print('Train and Test Folder already exists....!')
        print("I am skipping it....!")
    else:
        os.mkdir(dirr+'/'+'train')
        os.mkdir(dirr+'/'+'test')
        for i in range(cla):
            os.mkdir(dirr+'/'+'train'+'/'+'class_'+str(i))
            os.mkdir(dirr+'/'+'test'+'/'+'class_'+str(i))
        print('Folder Created Successfully....!')

################# CREATING FOLDER - END ################

if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    passed_args=args.parse_args()
    create_fold(config=passed_args.config)

