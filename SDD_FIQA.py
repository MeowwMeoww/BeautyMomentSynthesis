import os
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from misc.extract_bbox import *
from model import model
import numpy as np
import cv2
from config import *

device = config.DEVICE
model_path = CFG_FIQA.MODEL_PATH


def process_fiqa_image(img):
    data = torch.randn(1, 3, 112, 112)
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    data[0, :, :, :] = transform(img_pil)

    return data


def FIQA_network():
    model_path = CFG_FIQA.MODEL_PATH
    device = config.DEVICE
    network = model.R50([112, 112], use_type="Qua").to(device)
    net_dict = network.state_dict()
    data_dict = {
        key.replace('module.', ''): value for key, value in torch.load(model_path, map_location=device).items()}
    net_dict.update(data_dict)
    network.load_state_dict(net_dict)
    network.eval()

    return network


def FIQA(df, img_list, network):
    fiqa_scores = []
    keep_index = []

    for img_index in range(len(df)):
        input_data = get_target_bbox(img_list[img_index], df["bboxes"][img_index], p=CFG_FIQA.EXTEND_RATE)
        scores = []

        for cropped_face in input_data:
            img = process_fiqa_image(cropped_face).to(device)
            pred_score = network(img).data.cpu().numpy().squeeze()
            scores.append(pred_score)

        scores = [score.item() for score in scores]
        scores_avg = sum(scores)/len(scores)

        if scores_avg >= CFG_FIQA.THRESHOLD:
            keep_index.append(img_index)
            fiqa_scores.append([[score] for score in scores])

    new_df = df.iloc[keep_index]
    new_df['fiqa scores'] = fiqa_scores
    new_df = new_df.reset_index(drop=True)

    qualified_img = [img_list[index] for index in keep_index]
    qualified_img = np.array(qualified_img)

    return new_df, qualified_img
