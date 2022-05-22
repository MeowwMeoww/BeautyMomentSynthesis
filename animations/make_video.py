import os

from animations.animations import process_images_for_vid as process
from animations.animations import cover_animation as cover
from animations.animations import uncover_animation as uncover
from animations.animations import comb_animation as comb
from animations.animations import push_animation as push
from animations.animations import split_animation as split
from animations.animations import fade_animation as fade
from animations.animations import extract_vid as vid
from animations.animations import zoom_animation as zoom
from animations.animations import rotate_animation as rot
import pandas as pd
from misc.extract_bbox import zoom_rescale_bbox
from misc.extract_bbox import convert_bounding_box

import random
from tqdm import tqdm
import numpy as np
import shutil

def random_number():
    numb = random.randint(0, 5)
    return numb


def initialize_video(image, W, H, effect_speed, fps, duration):
    transitions = {
        0: cover,
        1: uncover,
        2: comb,
        3: push,
        4: split,
        5: fade,
    }

    '''to_bbox_animations = {
      0: zoom_in,
      1: rotate_zoom_in,
    }

    return_animation = {
      0: zoom_out,
      1: rotate_zoom_out,
    }'''
    
    black_screen = np.zeros([H, W, 3], dtype=np.uint8)
    black_screen.fill(0)  # black screen
    
    transitions[random_number()](img_list=[black_screen, image], 
                                               w=W, h=H,
                                               output_path="tmp/tmp_0.mp4",
                                               effect_speed=effect_speed, 
                                               fps=fps,
                                               duration=duration)
    

def find_inner_animation_images(df, names):
    best_fiqa_scores = dict.fromkeys(names)
    for name in names:
        best_fiqa_scores[name] = [-1, -1, -1]

    for idx in range(len(df)):
        for i in range(len(df["ids"][idx])):
            if int(df["fiqa scores"][idx][i][0]) > best_fiqa_scores[df["ids"][idx][i][0]][2]:
                best_fiqa_scores[df["ids"][idx][i][0]][1] = df["bboxes"][idx][i]
                best_fiqa_scores[df["ids"][idx][i][0]][0] = df["filename"][idx]
                best_fiqa_scores[df["ids"][idx][i][0]][2] = df["fiqa scores"][idx][i][0]

    filenames = []
    bboxes = []
    for i in best_fiqa_scores.values():
        filenames.append(i[0])
        bboxes.append(i[1])

    print(best_fiqa_scores)
    return filenames, bboxes


def make_video(info_df, names, img_list, output_path, effect_speed=1, duration=3, fps=30, fraction=1):
    animation = {
        0: cover,
        1: uncover,
        2: comb,
        3: push,
        4: split,
        5: fade,
    }

    inner_animation = {
        0: zoom,
        1: rot,
    }

    info_df = pd.DataFrame(data=info_df).reset_index(drop=True)

    os.mkdir("tmp")

    animated_filenames, bboxes = find_inner_animation_images(info_df, names)

    img_list, w, h = process(img_list, effect_speed, duration, fps, fraction=fraction)
    initialize_video(image=img_list[0], W=w, H=h, effect_speed=effect_speed, fps=fps, duration=duration)
    vid_paths = ["tmp/tmp_0.mp4"]
    
    for i in tqdm(range(len(img_list) - 1)):
        if info_df["filename"][i] in animated_filenames:
            idx = animated_filenames.index(info_df["filename"][i])
            s, bbox = zoom_rescale_bbox(coco_bbox=convert_bounding_box(box=bboxes[idx], input_type="opencv", change_to="coco"), W=w, H=h)
            inner_animation[random.randint(0, 1)](img=img_list[i], output_path="tmp/inner_{}.mp4".format(i), W=w, H=h, opencv_bbox=bbox, fps = fps, duration = int(2*duration/3))
            vid_paths.append("tmp/inner_{}.mp4".format(i))

        animation[random_number()](img_list=img_list[i:i + 2], w=w, h=h,
                                   output_path="tmp/tmp_{}.mp4".format(i+1), effect_speed=effect_speed, 
                                   fps=fps, duration=duration)
        vid_paths.append("tmp/tmp_{}.mp4".format(i+1))
    
    vid(vid_paths=vid_paths, output_path=output_path, w=w, h=h, fps=fps)
    shutil.rmtree("tmp")
