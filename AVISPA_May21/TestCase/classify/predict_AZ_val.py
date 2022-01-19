import time
import argparse
import os
import numpy as np
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
sys.path.append('../')
from classify import util
import config



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_img', dest='path_img', help='Path to image', required=True, default=None, type=str)
    parser.add_argument('--accuracy', action='store_false', help='To print accuracy score')
    parser.add_argument('--plot_confusion_matrix', action='store_false')
    parser.add_argument('--execution_time', action='store_true')
    parser.add_argument('--store_activations', action='store_false')
    parser.add_argument('--novelty_detection', action='store_true')
    parser.add_argument('--model', type=str, required=True, help='Base model architecture')
    parser.add_argument('--data_dir', help='Path to data train directory')
    parser.add_argument('--batch_size', default=500, type=int, help='How many files to predict on at once')
    parser.add_argument('--model_name', required=True)
    args = parser.parse_args()
    return args
	
	

def get_files(path_img):
    if os.path.isdir(path_img):
        files = glob.glob(os.path.join(path_img, '*.jpg'))
    elif path_img.find('*') > 0: 
        files = glob.glob(path_img)
    else: 
        files = [path_img]

    if not len(files):
        print('No images found by the given path')
        exit(1)

    return files
	
args = parse_args()

config.CLASSIFY['data_dir'] = args.data_dir
print(args.data_dir)
config.CLASSIFY['train_dir'] = os.path.join(args.data_dir, 'train/')
print(os.path.join(args.data_dir, 'train/'))
config.CLASSIFY['validation_dir'] = os.path.join(args.data_dir, 'validation/')
print(os.path.join(args.data_dir, 'validation/'))
config.CLASSIFY['model'] = args.model
print(args.model)
config.CLASSIFY['model_name'] = args.model_name
print(args.model_name)
util.set_img_format()
model_module = util.get_model_class_instance()
print(model_module)

if args.store_activations:print('y')
