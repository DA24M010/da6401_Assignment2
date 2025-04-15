import os
import glob
from collections import defaultdict
import pandas as pd

TRAIN_DATA_DIR = 'nature_12K/inaturalist_12K/train'
VAL_DATA_DIR = 'nature_12K/inaturalist_12K/val'

image_list = []

train_paths = {}
val_paths = {}

for label in os.listdir(TRAIN_DATA_DIR):
    cur_path = os.path.join(TRAIN_DATA_DIR, label)
    for filename in glob.glob(f'{cur_path}/*.jpg'):
        train_paths[filename] = label

for label in os.listdir(VAL_DATA_DIR):
    cur_path = os.path.join(VAL_DATA_DIR, label)
    for filename in glob.glob(f'{cur_path}/*.jpg'):
        val_paths[filename] = label

train_df = pd.DataFrame(list(train_paths.items()), columns= ["path", "label"])
val_df = pd.DataFrame(list(val_paths.items()), columns= ["path", "label"])
train_df.to_csv('train.csv', index = False)
val_df.to_csv('val.csv', index = False)