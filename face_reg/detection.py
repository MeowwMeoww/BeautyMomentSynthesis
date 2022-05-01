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


def read_image_from_path(path):
	img = cv2.imread(path, 1)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img


def padding_image(img, max_width, max_height):
	top = bottom = (max_height - img.shape[-3]) // 2
	left = right = (max_width - img.shape[-2]) // 2
	img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType = cv2.BORDER_CONSTANT, value = [255, 255, 255])
	img = cv2.resize(img, (max_width, max_height))  # cv2.resize(width, height)

	return img


def resize_images(img_list, purpose, fraction = 1):
	if purpose == 'anchor':
		max_width = max([img_list[i].shape[-2] for i in range(len(img_list))])
		max_height = max([img_list[i].shape[-3] for i in range(len(img_list))])
		len_lst = len(img_list)

		img_list = list(map(padding_image, img_list, [max_width] * len_lst, [max_height] * len_lst))
		img_list = np.stack(img_list, axis = 0)

	elif purpose == 'input':
		all_width = [img_list[i].shape[-2] for i in range(len(img_list))]
		mean_width = int(sum(all_width) / len(all_width) * fraction)
		all_height = [img_list[i].shape[-3] for i in range(len(img_list))]
		mean_height = int(sum(all_height) / len(all_height) * fraction)

		img_list = np.stack([cv2.resize(img_list[i], (mean_width, mean_height)) for i in range(len(img_list))], axis = 0)

	return img_list


def return_paths(root, purpose, batch_size = 32):
	all_paths = [join(path, name) for path, _, files in os.walk(root) for name in files if os.path.isfile(join(path, name))]
	paths = [join(path, name) for path, _, files in os.walk(root) for name in files if os.path.isfile(join(path, name))]

	if purpose == 'input':
		labels = [path.replace('\\', '/').rsplit('/')[-1] for path in paths]

		labels = [labels[x:x + batch_size] for x in range(0, len(labels), batch_size)]
		paths = [paths[x:x + batch_size] for x in range(0, len(paths), batch_size)]

	elif purpose == 'anchor':
		labels = [path.replace('\\', '/').rsplit('/')[-2] for path in paths]

	return paths, labels


def read_images(paths, purpose):
	img_list = list(map(read_image_from_path, paths))
	shape_check = all(img_list[i].shape == img_list[0].shape for i in range(len(img_list)))
	if shape_check:
		img_list = np.array(img_list)
	else:
		img_list = resize_images(img_list, purpose)

	return img_list


def create_facenet_models():
	"""
	This function returns an MTCNN + InceptionResnet V1 model bases - which was used to detect and encode human
	faces in images.
	Original GitHub Repository: https://github.com/timesler/facenet-pytorch

	To have a better understanding of this model's parameters,
	use the Python built-in help () function
	>> help (mtcnn_model_name)
	Example
	>> model_A = create_mtcnn_model()
	>> help (model_A)

	To calibrate again MTCNN model's parameters after calling out this function:
	>> mtcnn_model_name.parameters_want_to_change = ...
	Example
	>> model_A = create_mtcnn_model()
	>> model_A.image_size = 200
	>> model_A.min_face_size = 10
"""

	device = config.DEVICE
	infer_model = InceptionResnetV1(pretrained = 'vggface2', device = device).eval()

	mtcnn = MTCNN(
		image_size = 160, margin = 0, min_face_size = 50,
		thresholds = [0.7, 0.7, 0.8], post_process = False,
		device = device, selection_method = 'largest_over_threshold'
	)

	mtcnn.keep_all = False

	return mtcnn, infer_model


def get_bounding_box(mtcnn_model, frames, batch_size = 32):
	"""
This function detects human faces in the given batch of images / video frames
in the Python Numpy Array format. It will return 3 lists - bounding box coordinates
list, confidence score list, and facial landmarks list. See details on Return section.

To save GPU's memory, this function will detect faces in
separate mini-batches. It has been shown that mini-batched detection has the
same efficiency as full-batched detection.

For each detected face, the function will return:
* 4 bounding box coordinates (x left, y top, x right, y bot)
* 1 confidence score for that bounding box.
* 5 landmarks - marking that person's eyes, nose, and mouth.

Parameters
----------
+ mtcnn_model: a facenet_pytorch.models.mtcnn.MTCNN model.
		Passing a MTCNN model base that has been created beforehand.

+ frames: np.ndarray.
		Given batch of images to detect human faces. Must be a Numpy Array that
		has 4D shape.
		--> The ideal shape should be:
		(number_of_samples, image_width, image_height, channels)

		All images in the Frames must be of equal size, and has all pixel values
		in scale [0-255].
		All images must have 3 color channels (RGB-formatted images).

+ batch_size: int > 0, optional, default is 32.
		The size of the mini-batch. The larger the batch size, the more GPU memory
		needed for detection.

Return
----------
+ bboxes_pred_list: list .
		The list that contains all the predicted bounding boxes in the OpenCV format
		[x_left, y_top, x_right, y_bot]

+ box_probs_list: list .
		The list that contains the confidence scores for all predicted bounding
		boxes.

+ landmark_list: list .
		The list that contains facial landmarks for all predicted bounding boxes/
"""

	assert (type(frames) == np.ndarray and frames.ndim == 4), "Frames must be a 4D np.array"
	assert (frames.shape[-1] == 3), "All images must have 3 color channels - R, G, and B"
	assert (type(batch_size) == int and batch_size > 0), "Batch size must be an integer number, larger than 0"

	size_checking = all(frame.shape == frames[0].shape for frame in frames)
	assert size_checking, "All the images must be of same size"

	frames = frames.astype(np.uint8)
	steps = math.ceil(len(frames) / batch_size)
	frames = np.array_split(frames, steps)

	bboxes_pred_list = []
	box_probs_list = []
	landmark_list = []

	for batch_file in frames:
		with torch.no_grad():
			bb_frames, box_probs, landmark = mtcnn_model.detect(batch_file, landmarks = True)

		for ind in range(len(bb_frames)):
			if bb_frames[ind] is not None:
				bboxes_pred_list.append(bb_frames[ind].tolist())
				box_probs_list.append(box_probs[ind].tolist())
				landmark_list.append(landmark[ind].tolist())

			else:
				bboxes_pred_list.append([None])
				box_probs_list.append([None])
				landmark_list.append([None])

	return bboxes_pred_list, box_probs_list, landmark_list


def convert_bounding_box(box, input_type, change_to):
	"""
	This function converts an input bounding box to either YOLO, COCO, or OpenCV
	format.
	However, the function only converts the input bounding box if it already belongs
	to one of the three formats listed above.

	Note:
	+ OpenCV-formatted bounding box has 4 elements [x_left, y_top, x_right, y_bot]
	+ YOLO-formatted bounding box has 4 elements [x_center, y_center, width, height]
	+ COCO-formatted bounding box has 4 elements [x_left, y_top, width, height]

	Parameters
	----------
	+ box : list.
			The provided bounding box in the Python list format. The given bounding box
			must have 4 elements, corresponding to its format.

	+ input_type : {'opencv', 'yolo', 'coco'}
			The format of the input bounding box.
			Supported values are 'yolo' - for YOLO format, 'coco' - for COCO format,
			and 'opencv' - for OpenCV format.

	+ change_to : {'opencv', 'yolo', 'coco'}.
			The type of format to convert the input bounding box to.
			Supported values are 'yolo' - for YOLO format, 'coco' - for COCO format,
			and 'opencv' - for OpenCV format.

	Return
	----------
			Returns a list for the converted bounding box.

	"""
	assert (type(box) == list), 'The provided bounding box must be a Python list'
	assert (len(box) == 4), 'Must be a bounding box that has 4 elements (OpenCV format)'
	assert (input_type == 'yolo' or input_type == 'coco' or input_type == 'opencv')
	assert (change_to == 'yolo' or change_to == 'coco' or change_to == 'opencv')
	assert (input_type != change_to), "The format between your input bounding boxes nad output boxes must be different."

	if input_type == 'opencv':
		x_left, y_top, x_right, y_bot = box[0], box[1], box[2], box[3]

		if change_to == 'yolo':
			x_center = int((x_left + x_right) / 2)
			y_center = int((y_top + y_bot) / 2)
			width = int(x_right - x_left)
			height = int(y_bot - y_top)

			return [x_center, y_center, width, height]

		elif change_to == 'coco':
			width = int(x_right - x_left)
			height = int(y_bot - y_top)

			return [x_left, y_top, width, height]

	elif input_type == 'yolo':
		x_center, y_center, width, height = box[0], box[1], box[2], box[3]

		if change_to == 'opencv':
			x_left = int(x_center - width / 2)
			x_right = int(x_center + width / 2)
			y_top = int(y_center - height / 2)
			y_bot = int(y_center + height / 2)

			return [x_left, x_right, y_top, y_bot]

		elif change_to == 'coco':
			x_left = int(x_center - width / 2)
			y_top = int(y_center - height / 2)

			return [x_left, y_top, width, height]

	elif input_type == 'coco':
		x_left, y_top, width, height = box[0], box[1], box[2], box[3]

		if change_to == 'opencv':
			x_right = int(x_left + width)
			y_bot = int(y_top + height)

			return [x_left, x_right, y_top, y_bot]

		elif change_to == 'yolo':
			x_center = int(x_left + width / 2)
			y_center = int(y_top + height / 2)

			return [x_center, y_center, width, height]


def fixed_image_standardization(image_tensor):
	processed_tensor = (image_tensor - 127.5) / 128.0
	return processed_tensor


def transform(img):
	normalized = transforms.Compose([
		transforms.ToTensor(),
		fixed_image_standardization
	])
	return normalized(img)


def filter_images(name, img_list, boxes, paths):
	keep_index = [i for i, x in enumerate(boxes) if x[0] != [None]]

	img_final = [img_list[i] for i in keep_index]
	paths_final = [paths[i] for i in keep_index]
	box_list_final = [boxes[i] for i in keep_index]
	name_final = [name[i] for i in keep_index]

	return np.array(img_final), box_list_final, name_final, paths_final


def clipping_boxes(img_list, boxes):
	def clipping_method(img, box, format = 'opencv'):
		if format == 'opencv':
			x_left, y_top, x_right, y_bot = int(box[0]), int(box[1]), int(box[2]), int(box[3])

			x_left = min(max(x_left, 0), img.shape[-2])
			y_top = min(max(y_top, 0), img.shape[-3])
			x_right = min(max(x_right, 0), img.shape[-2])
			y_bot = min(max(y_bot, 0), img.shape[-3])

			return [x_left, y_top, x_right, y_bot]

		elif format == 'coco':
			x_left, y_top, width, height = int(box[0]), int(box[1]), int(box[2]), int(box[3])

			x_left = min(max(x_left, 0), img.shape[-3])
			y_top = min(max(y_top, 0), img.shape[-2])
			width = min(max(width, 0), img_list.shape[-3] - x_left)
			height = min(max(height, 0), img_list.shape[-2] - y_top)

			return [x_left, y_top, width, height]

		elif format == 'yolo':
			x_center, y_center, width, height = int(box[0]), int(box[1]), int(box[2]), int(box[3])

			x_center = min(max(x_center, 0), img.shape[0])
			y_center = min(max(y_center, 0), img.shape[0])
			width = min(max(width, 0), img_list.shape[0])
			height = min(max(height, 0), img_list.shape[1])

			return [x_center, y_center, width, height]

	box_clipping = []

	for i in range(len(img_list)):
		if len(boxes[i]) >= 1 and boxes[i][0] is not None:
			img_list_map = np.expand_dims(img_list[i], axis = 0)
			img_list_map = np.repeat(img_list_map, repeats = len(boxes[i]), axis = 0)
			box_clipping.append(list(map(clipping_method, img_list_map, boxes[i])))
		else:
			box_clipping.append([[None]])

	return box_clipping


def cropping_face(img_list, box_clipping, percent = CFG_REG.CROP.EXTEND_RATE, purpose = None):
	def crop_with_percent(img, box, rate = 0):
		x_left, y_top, x_right, y_bot = box[0], box[1], box[2], box[3]  # [x_left, y_top, x_right, y_bot]

		x_left -= rate * (x_right - x_left)
		x_right += rate * (x_right - x_left)
		y_top -= rate * (y_bot - y_top)
		y_bot += rate * (y_bot - y_top)
		target_img = img[int(y_top): int(y_bot), int(x_left): int(x_right)]

		target_img = cv2.resize(target_img, tuple(CFG_REG.CROP.FACE_SIZE),
														interpolation = cv2.INTER_CUBIC)  # cv2 resize (height, width)

		return np.array(target_img).astype('int16')

	if purpose == 'input':
		cropped_faces = []
		for i in range(len(img_list)):
			if len(box_clipping[i]) >= 1:
				img_list_map = np.expand_dims(img_list[i], axis = 0)
				img_list_map = np.repeat(img_list_map, repeats = len(box_clipping[i]), axis = 0)
				cropped_faces.append(list(map(crop_with_percent, img_list_map, box_clipping[i])))

	elif purpose == 'anchor':
		cropped_faces = [crop_with_percent(img_list[i], box_clipping[i][0], percent) for i in range(len(img_list))]  # cần fix lại

	return cropped_faces


def vector_embedding(infer_model, img_list, purpose = 'input'):
	device = config.DEVICE

	def extract_vector(batch):
		batch = batch.to(device)
		embed = infer_model(batch)
		embed = embed.cpu().detach().numpy()
		return embed

	if purpose == 'anchor':
		img_list = list(map(transform, img_list))
		img_list = torch.stack(img_list, dim = 0)

		batch_size = CFG_REG.BATCH_SIZE
		steps = math.ceil(len(img_list) / batch_size)
		img_list = torch.split(img_list, steps)

		vector_embeddings = list(map(extract_vector, img_list))
		vector_embeddings = np.concatenate(vector_embeddings).reshape(-1, 512)

	elif purpose == 'input':
		vector_embeddings = []

		for img in img_list:
			mini_list = list(map(transform, img))
			mini_list = torch.stack(mini_list, dim = 0)
			embedding = extract_vector(mini_list)

			if embedding.shape[0] > 1:
				embedding = np.concatenate(embedding).reshape(-1, 512)

			embedding = list(embedding)

			vector_embeddings.append(embedding)

	return vector_embeddings


def names_to_integers(list_name):
	unique_names = np.unique(list_name)
	label_to_int = {label: i for i, label in enumerate(unique_names)}
	int_to_label = {i: label for i, label in enumerate(unique_names)}
	mapped_name = np.array([label_to_int[name] for name in list_name]).astype('int16')
	return int_to_label, mapped_name


def euclidean_distance(row1, row2):
	euclidean_dist = norm(row1 - row2[:-1])

	return (euclidean_dist, int(row2[-1]))


def cosine_distance(row1, row2):
	return dot(row1, row2) / (norm(row1) * norm(row2))


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	test_rows = np.array([test_row] * len(train))
	euclidean_distances = list(map(euclidean_distance, test_rows, train))
	euclidean_distance_index = euclidean_distances.copy()
	euclidean_distance_index = sorted(range(len(euclidean_distance_index)),
									                        key = lambda tup: euclidean_distance_index[tup])
	euclidean_distances.sort(key = lambda tup: tup[0])
	neighbors = list()
	cosine_scores = list()
	for i in range(num_neighbors):
		cos_dist = cosine_distance(test_row, train[euclidean_distance_index[i]][:-1])
		if cos_dist < CFG_REG.KNN.THRESHOLD:
			neighbors.append(None)
			cosine_scores.append(None)
		else:
			neighbors.append(euclidean_distances[i][1])
			cosine_scores.append(cos_dist)
	return neighbors, cosine_scores


# Make a prediction with neighbors
def classification(mapping, train, test_row, num_neighbors):
	neighbors, cosine_scores = get_neighbors(train, test_row, num_neighbors)
	output_values = [row for row in neighbors if row is not None]
	if output_values:
		prediction = max(set(output_values), key = output_values.count)
		prediction_index = [i for i in range(len(output_values)) if output_values[i] == prediction]
		cosine_score = max([cosine_scores[i] for i in prediction_index])
		prediction = mapping[prediction]
	else:
		prediction = None
		cosine_score = None

	return [prediction], [cosine_score]


# KNN Algorithm
def k_nearest_neighbors(label, train, test, num_neighbors):
	int_to_label, anchor_mapped_label = names_to_integers(label)

	new_shape = list(train.shape)
	new_shape[-1] += 1
	new_shape = tuple(new_shape)

	anchor_mega = np.empty(new_shape)
	for i in range(len(anchor_mega)):
		anchor_mega[i] = np.append(train[i], anchor_mapped_label[i])

	predictions = list()
	cosine_prediction = list()
	for row in test:
		output, score = classification(int_to_label, anchor_mega, row, num_neighbors)
		predictions.append(output)
		cosine_prediction.append(score)

	return predictions, cosine_prediction


def knn_prediction(anchor_label, anchor_embed, input_embed):
	predicted_ids, predicted_scores = map(list, zip(*[k_nearest_neighbors(anchor_label, anchor_embed, embed,
																		  CFG_REG.KNN.NUM_NEIGHBORS) for embed in input_embed]))
	# list comprehension returns multiple lists

	return predicted_ids, predicted_scores


def indices(seq, values):
	matched_index = []
	for value in values:
		lst = [i for i, x in enumerate(seq) if x == value]
		matched_index.append(lst)

	return matched_index


def check_duplicates_ids(ids_list, scores_list, bbox_list):
	check = [ids[0] for ids in ids_list]
	if len(check) != len(set(check)):
		indices_list = indices(check, (key for key, count in Counter(check).items() if count > 1))
		non_rep = indices(check, (key for key, count in Counter(check).items() if count == 1))
		non_rep = [item for sublist in non_rep for item in sublist]
		keep_id = list()
		keep_id += non_rep

		for img_id in indices_list:
			scores = [scores_list[i] for i in img_id]
			scores = np.array(scores)
			max_id = np.argmax(scores)
			keep_id.append(img_id[max_id])

		cleared_bbox = [bbox_list[i] for i in keep_id]
		cleared_scores = [scores_list[i] for i in keep_id]
		cleared_ids = [ids_list[i] for i in keep_id]

	else:
		cleared_bbox = bbox_list
		cleared_scores = scores_list
		cleared_ids = ids_list

	return cleared_bbox, cleared_scores, cleared_ids


def clear_results(images, scores, img_names, boxes, ids, paths, person = None):
	keep_img = list()

	if person:
		for i in range(len(ids)):
			keep_index = [ind for ind in range(len(ids[i])) if ids[i][ind][0] in person]
			boxes[i] = [boxes[i][ind] for ind in keep_index]
			ids[i] = [ids[i][ind] for ind in keep_index]
			scores[i] = [scores[i][ind] for ind in keep_index]

			if keep_index:
				keep_img.append(i)

	else:
		for i in range(len(ids)):
			keep_index = [ind for ind in range(len(ids[i])) if ids[i][ind] != [None]]
			boxes[i] = [boxes[i][ind] for ind in keep_index]
			ids[i] = [ids[i][ind] for ind in keep_index]
			scores[i] = [scores[i][ind] for ind in keep_index]

			if keep_index:
				keep_img.append(i)

	new_names = [img_names[i] for i in range(len(img_names)) if boxes[i]]
	new_scores = list(filter(None, scores))
	new_boxes = list(filter(None, boxes))
	new_ids = list(filter(None, ids))

	new_boxes, new_scores, new_ids = map(list, (zip(*map(check_duplicates_ids, new_ids, new_scores, new_boxes))))
	images = [images[i] for i in keep_img]
	paths = [paths[i] for i in keep_img]

	df_new = pd.DataFrame({'filename': new_names, 'bboxes': new_boxes, 'ids': new_ids, 'face scores': new_scores, 'paths': paths})
	df_new = df_new.reset_index(drop = True)

	return df_new, np.array(images)


def face_detection(input_paths, input_names, anchor_paths, anchor_labels, mtcnn, infer_model, finding_name):
	"""
	This function performs face detection in the given image dataset.

	Parameters
	----------
	+ original_path : str.
	The path to your input image dataset.

	+ anchor_path : str.
	The path to your anchor image dataset.

	+ finding_name: list.
	A list of names of people we need to find.

	Return
	----------
	+ df : Pandas Dataframe.
	A dataframe contained the filenames for input images, as well as predicted bounding boxes.

	+ input_img: np.ndarray.
	"""

	input_img = read_images(input_paths, purpose = 'input')
	anchor_img = read_images(anchor_paths, purpose = 'anchor')

	mtcnn, infer_model = create_facenet_models()

	input_boxes, _, _ = get_bounding_box(mtcnn, input_img, CFG_REG.BATCH_SIZE)
	anchor_boxes, _, _ = get_bounding_box(mtcnn, anchor_img, CFG_REG.BATCH_SIZE)

	input_boxes = clipping_boxes(input_img, input_boxes)
	anchor_boxes = clipping_boxes(anchor_img, anchor_boxes)

	input_img, input_boxes, input_names, input_paths = filter_images(input_names, input_img, input_boxes, input_paths)
	anchor_img, anchor_boxes, anchor_label, anchor_paths = filter_images(anchor_labels, anchor_img, anchor_boxes, anchor_paths)

	cropped_img_anchor = cropping_face(anchor_img, anchor_boxes, purpose = 'anchor')
	cropped_img_input = cropping_face(input_img, input_boxes, purpose = 'input')

	anchor_embed = vector_embedding(infer_model, cropped_img_anchor, purpose = 'anchor')
	input_embed = vector_embedding(infer_model, cropped_img_input, purpose = 'input')

	final_ids, final_scores = knn_prediction(anchor_label, anchor_embed, input_embed)

	df, input_img = clear_results(images = input_img, img_names = input_names, scores = final_scores,
																boxes = input_boxes, ids = final_ids, paths = input_paths, person = finding_name)

	return df, input_img
