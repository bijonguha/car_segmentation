#Default Packages
import os
import sys
import random
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Installed Packages
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Custom Packages
# Root directory of the project
ROOT_DIR = os.path.abspath("")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'model', 'mask_rcnn_coco.h5')

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def image_seg(file_names):
  '''
  Function for car image segmentation
  
  filename : str
    name of the Image
  
  Returns
    None
  '''
  # Load a random image from the images folder
  # file_names = next(os.walk(IMAGE_DIR))[2]
  image = skimage.io.imread(file_names)

  # Run detection
  results = model.detect([image], verbose=0)

  idx_ar = np.where(results[0]['class_ids'] == 3)

  for i,idx in enumerate(idx_ar[0]):
    if i == 0:
      car_mask = results[0]['masks'][:,:,idx]
    else:
      car_mask = car_mask + results[0]['masks'][:,:,idx]

  tmp, tmp_b = image.copy(), image.copy()
  tmp[~car_mask] = 0
  tmp_b[car_mask] = 0

  img_overlay = cv2.addWeighted(tmp,1,tmp_b,0.35,0)
  
  plt.rcParams["figure.figsize"] = (16,10)
  plt.imshow(np.hstack((image,img_overlay)))
  plt.show()

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='FcarScan Task - Car Segmentation and Background removal')

  parser.add_argument('--image', default='view1.jpeg', type=str, help='Path of the image file')
  parser.add_argument('--folder', default='False', type=str, help='True if path is a directory')

  args = parser.parse_args()

  if args.folder == 'True':
    import os
    files = os.listdir(args.image)
    files = [os.path.join(args.image, file) for file in files]
    for file in files:
      print('Image name :', file)
      image_seg(file)

  else:
    print('Image name :', args.image)
    image_seg(args.image)