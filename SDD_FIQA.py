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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = CFG_FIQA.MODEL_PATH


def process_fiqa_image(img):

    data = torch.randn(1, 3, 112, 112)

    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    data[0, :, :, :] = transform(im_pil)

    return data


def network(model_path, device):
    net = model.R50([112, 112], use_type="Qua").to(device)
    net_dict = net.state_dict()
    data_dict = {
        key.replace('module.', ''): value for key, value in torch.load(model_path, map_location=device).items()}
    net_dict.update(data_dict)
    net.load_state_dict(net_dict)
    net.eval()

    return net


def FIQA(df, img_list):
    net = network(model_path, device)
    fiqa_scores = []
    keep_index = []

    for i in range(len(df)):
        input_data = get_target_bbox(img_list[i], df["bboxes"][i], p=0.15)
        score = []
        for j in input_data:
            if j.shape[0] > 0 and j.shape[1] > 0:
                img = process_fiqa_image(j).to(device)
                pred_score = net(img).data.cpu().numpy().squeeze()
                score.append(pred_score)

            if max(score) > config.FIQA.THRESHOLD:
                keep_index.append(i)
                fiqa_scores.append(score[0].item())

    new_df = df.iloc[keep_index]
    new_df['fiqa scores'] = fiqa_scores
    new_df = new_df.reset_index(drop=True)

    qualified_img = [img_list[index] for index in keep_index]
    qualified_img = np.array(qualified_img)

    return new_df, qualified_img
