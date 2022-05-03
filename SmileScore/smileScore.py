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
	final_img = []
	smile_scores = []

	for i in range(len(df)):
		input_data = get_target_bbox(img_list[i], df["bboxes"][i], p = CFG_SMILE.EXTEND_RATE)
		scores = []
		for cropped_face in input_data:
			cropped_face = cropped_face[..., ::-1].copy()

			try:
				predictions = DeepFace.analyze(cropped_face, actions = ['emotion'], detector_backend = 'mtcnn', enforce_detection = True)
			except:
				predictions = DeepFace.analyze(cropped_face, actions = ['emotion'], detector_backend = 'retinaface', enforce_detection = True)

			scores.append(predictions['emotion']['happy'])

		smile_scores.append([[score] for score in scores])
		smile_score_avg.append(sum(scores) / len(scores))
		final_img.append(img_list[i])

	new_df = df.copy()
	new_df['smile score average'] = smile_score_avg
	new_df['smile scores'] = smile_scores
	new_df.sort_values(by = 'smile score average', ascending = False, inplace = True)
	old_index = list(new_df.index)
	final_img = [final_img[i] for i in old_index]

	new_df.reset_index(drop = True)
	return new_df, np.array(final_img)
