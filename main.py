import base64
import io
import keras.engine.topology as KE
import tf_slim as slim
import numpy as np
import shapely
from shapely.geometry import polygon as shapely_poly
from shapely.geometry import box
import argparse
import pickle
import pathlib
from pathlib import Path
import mrcnn
from mrcnn.model import MaskRCNN
import mrcnn.utils
import mrcnn.config
import cv2
import git
import os
if not os.path.exists("Parking Space"):
    print("Clonning M_RCNN repository...")
    git.Git("./").clone("https://github.com/Jarvis109/Parking-Space-detector.git")
class Config(mrcnn.config.Config):
    NAME = "Model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 81
config = Config()
config.display()
Root_dir = os.getcwd()
MODEL_DIR = os.path.join(Root_dir, "logs")
coco_model_path = os.path.join(Root_dir, "mask_rcnn_coco.h5")
print(coco_model_path)
if not os.path.exists(coco_model_path):
    mrcnn.utils.download_trained_weights(coco_model_path)
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config = Config())
model.load_weights(coco_model_path, by_name=True)
def get_cars(boxes, class_ids):
    cars = []
    for i, box in enumerate(boxes):
        if class_ids[i] in [3, 8, 6]:
            cars.append(box)
    return np.array(cars)





