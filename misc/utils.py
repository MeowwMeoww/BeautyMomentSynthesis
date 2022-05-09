from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import transforms
import numpy as np
import cv2
import pandas as pd
import math
import os
from os.path import join
from sklearn.neighbors import KNeighborsClassifier
from numpy import dot
from numpy.linalg import norm
from collections import Counter
from config import *
from misc.utils import *

def check_ids_equal(list_A, list_B):
    flatten_A = [x for x in list_A]
    flatten_B = [x for x in list_B]

    return sorted(flatten_A) == sorted(flatten_B)

def rescale_box_with_img_ratio(bbox, before_rescale_img, after_rescale_img):
  '''
  Note: the bbox will be converted from the OPENCV format to the YOLO format 
  '''
  x_left, y_top, x_right, y_bot = bbox

  old_img_height, old_img_width = before_rescale_img.shape[-3:-1]
  new_img_height, new_img_width = after_rescale_img.shape[-3:-1]
  width_change_ratio = new_img_width / old_img_width
  height_change_ratio = new_img_height / old_img_height

  x_left = x_left * width_change_ratio
  x_right = x_right * width_change_ratio
  y_top = y_top * height_change_ratio
  y_bot = y_bot * height_change_ratio

  return [int(x_left), int(y_top), int(x_right), int(y_bot)]

def rescale_bboxes(bbox_list, before_img_list, after_img_list):
    box_original_size = []

    for img_index in range(len(before_img_list)):
        if len(bbox_list[img_index]) >= 1 and bbox_list[img_index][0] != [None]:
            before_img_map = np.expand_dims(before_img_list[img_index], axis = 0)
            before_img_map = np.repeat(before_img_map, repeats = len(bbox_list[img_index]), axis = 0)
            after_img_map = np.expand_dims(after_img_list[img_index], axis = 0)
            after_img_map = np.repeat(after_img_map, repeats = len(bbox_list[img_index]), axis = 0)
            box_original_size.append(list(map(rescale_box_with_img_ratio, bbox_list[img_index], before_img_map, after_img_map)))
        else:
            box_original_size.append([[None]])

    return box_original_size