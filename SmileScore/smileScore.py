import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
from misc.extract_bbox import *
from config import *
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace


def get_smile_score(df, img_list):
    smile_score_avg = []
    smile_scores = []

    for img_index in range(len(df)):
        input_data = get_target_bbox(img_list[img_index], df["bboxes"][img_index], p = CFG_SMILE.EXTEND_RATE)
        faces_smile_scores = []

        for cropped_face in input_data:
            cropped_face = cropped_face[..., ::-1].copy() #RGB --> BGR

            try:
                predictions = DeepFace.analyze(cropped_face, actions = ['emotion'], detector_backend = 'mtcnn', enforce_detection = True)
            except:
                predictions = DeepFace.analyze(cropped_face, actions = ['emotion'], detector_backend = 'retinaface', enforce_detection = True)

            faces_smile_scores.append(predictions['emotion']['happy'])

        smile_scores.append([[score] for score in faces_smile_scores])
        smile_score_avg.append(sum(faces_smile_scores) / len(faces_smile_scores))

    new_df = df.copy()
    new_df['smile score average'] = smile_score_avg
    new_df['smile scores'] = smile_scores
    new_df.sort_values(by = 'smile score average', ascending = False, inplace = True)
    old_index = list(new_df.index)
    final_img = [img_list[index] for index in old_index]

    new_df.reset_index(drop = True)
    return new_df, np.array(final_img)
