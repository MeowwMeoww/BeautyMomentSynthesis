import os
from animations.animations import process_images_for_vid as process
from animations.animations import cover_animation as cover
from animations.animations import uncover_animation as uncover
from animations.animations import comb_animation as comb
from animations.animations import push_animation as push
from animations.animations import split_animation as split
from animations.animations import fade_animation as fade
from animations.animations import extract_vid as vid
import random
from tqdm import tqdm
import numpy as np


def random_number():
    numb = random.randint(0, 1000)
    return numb % 6	


def initialize_video(image, W, H, effect_speed, fps, duration):
    animation = {
        0: cover,
        1: uncover,
        2: comb,
        3: push,
        4: split,
        5: fade,
    }
    
    black_screen = np.zeros([H, W, 3], dtype=np.uint8)
    black_screen.fill(0)  # black screen
    
    frames = []
    frames.append(animation[random_number()](img_list=[black_screen, image], 
                                             w=W, h=H,
                                             effect_speed=effect_speed, 
                                             fps=fps,
                                             duration=duration)
                      )
    return frames
    

def make_video(img_list, output_path, effect_speed=1, duration=3, fps=30, fraction=1):
    animation = {
        0: cover,
        1: uncover,
        2: comb,
        3: push,
        4: split,
        5: fade,
    }

    img_list, w, h = process(img_list, effect_speed, duration, fps, fraction=fraction)
    frames = initialize_video(image=img_list[0], W=w, H=h, effect_speed=effect_speed, fps=fps, duration=duration)    
    
    for i in tqdm(range(len(img_list) - 1)):
        frames.append(animation[random_number()](img_list=img_list[i:i + 2], w=w, h=h,
                                                 effect_speed=effect_speed, fps=fps,
                                                 duration=duration)
                      )
    
    vid(frames=frames, output_path=output_path, w=w, h=h, fps=fps)


# img_list = []
# for i in range(len(os.listdir(r"test/img"))):
    # img_list.append("test/img/"+os.listdir(r"test/img")[i])
# make_video(img_list=img_list, output_path="results/output_test.avi", effect_speed=2, duration=5, fps=120, fraction=2)
