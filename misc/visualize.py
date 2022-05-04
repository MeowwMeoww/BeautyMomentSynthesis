import cv2
import numpy as np
from config import *


def drawing_boxes(row, img):
    row = row[1]

    img_name = row['filename']
    fiqa_scores = row['fiqa scores']
    smile_scores = row['smile scores']
    bboxes = row['bboxes']
    face_scores = row['face scores']
    ids = row['ids']

    img = cv2.putText(img, text = 'Image name: {}'.format(img_name), org = tuple(VISUALIZE.HEADER.ORG),
                      fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = VISUALIZE.HEADER.FONT_SCALE,
                      color = VISUALIZE.HEADER.COLOR, thickness = VISUALIZE.HEADER.THICKNESS, lineType = cv2.LINE_AA)

    for bbox_index in range(len(bboxes)):
        current_box = bboxes[bbox_index]

        start_point = (current_box[0], current_box[1])
        end_point = (current_box[2], current_box[3])

        img = cv2.rectangle(img, start_point, end_point, VISUALIZE.BBOX.COLOR, thickness = VISUALIZE.BBOX.THICKNESS)

        img = cv2.rectangle(img, start_point, end_point, VISUALIZE.BBOX.COLOR, thickness = VISUALIZE.BBOX.THICKNESS)

        text_size, _ = cv2.getTextSize('Face: {0:.3g}'.format(face_scores[bbox_index][0]), cv2.FONT_HERSHEY_SIMPLEX, VISUALIZE.NOTATIONS.FONT_SCALE,
                                        VISUALIZE.NOTATIONS.THICKNESS)

        text_w, text_h = text_size

        img = cv2.rectangle(img, (start_point[0], start_point[1] - 4 * text_h - 2 * VISUALIZE.NOTATIONS.SPACE),
                            (start_point[0] + text_w + VISUALIZE.BACKGROUND.WIDTH_EXTENSION, start_point[1] - VISUALIZE.NOTATIONS.SPACE), VISUALIZE.BACKGROUND.COLOR, -1)

        img = cv2.putText(img, text = 'Face: {0:.3g}'.format(face_scores[bbox_index][0]),
                          org = (start_point[0], start_point[1] - 3 * VISUALIZE.NOTATIONS.SPACE),
                          fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = VISUALIZE.NOTATIONS.FONT_SCALE, color = VISUALIZE.NOTATIONS.COLOR,
                          thickness = VISUALIZE.NOTATIONS.THICKNESS, lineType = cv2.LINE_AA)

        img = cv2.putText(img, text = 'FIQA: {0:.3g}'.format(fiqa_scores[bbox_index][0]),
                          org = (start_point[0], start_point[1] - 2 * VISUALIZE.NOTATIONS.SPACE),
                          fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = VISUALIZE.NOTATIONS.FONT_SCALE, color = VISUALIZE.NOTATIONS.COLOR,
                          thickness = VISUALIZE.NOTATIONS.THICKNESS, lineType = cv2.LINE_AA)

        img = cv2.putText(img, text = 'Smile: {0:.3g}'.format(smile_scores[bbox_index][0]),
                          org = (start_point[0], start_point[1] - VISUALIZE.NOTATIONS.SPACE),
                          fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = VISUALIZE.NOTATIONS.FONT_SCALE, color = VISUALIZE.NOTATIONS.COLOR,
                          thickness = VISUALIZE.NOTATIONS.THICKNESS, lineType = cv2.LINE_AA)

        img = cv2.putText(img, text = '{}'.format(ids[bbox_index][0]), org = (start_point[0], start_point[1] - 4 * VISUALIZE.NOTATIONS.SPACE),
                          fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = VISUALIZE.NOTATIONS.FONT_SCALE, color = VISUALIZE.NOTATIONS.COLOR,
                          thickness = VISUALIZE.NOTATIONS.THICKNESS, lineType = cv2.LINE_AA)

    return img


def visualizing_bounding_boxes(df, img_list):
    img_list = list(map(drawing_boxes, list(df.iterrows()), img_list))
    img_list = np.array(img_list)

    return img_list

