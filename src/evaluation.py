from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import argparse
from get_data import get_data, read_params
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def evaluate(config_file):
    config= get_data(config_file)
    batch = config['img_augment']['batch_size']
    class_mode = config['img_augment']['class_mode']
    test_pat = config['model']['test_path']
    model = load_model('model/sav_dir/trained.h5')
    config = get_data(config_file)

    test_gen = ImageDataGenerator(rescale=1./255)
    test_set = test_gen.flow_from_directory(test_gen, target_size=(255,255), batch_size=batch, class_mode=class_mode)

    label_map = (test_set.class_indices)
    #print(label_map)

    y_pred = model.predict(test_set)
    y_pred = np.argmax(y_pred, axis=1)

    print("confusin Matrix")
    sns.heatmap(confusion_matrix(test_set.classes, y_pred))
    plt.xlable("Actual Values")
    plt.ylabel("Predicted Value")
    plt.savefig('reports/confusion_matrix.png')
    #plt.show()

    print("Classification Report")
    target_names = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
    df = pd.DataFrame(classification_report(test_set.classes, y_pred, target_names=target_names, output_dict=True))
    df.to_csv('reports/classification_report.csv')



if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    passed_args=args.parse_args()
    evaluate(config=passed_args.config)