import random
import cv2
from tqdm import tqdm
import numpy as np


def process_images_for_vid(img_list, effect_speed, duration, fps, fraction=1):
    h = []
    w = []

    for image in img_list:
        height, width, _ = image.shape
        h.append(height)
        w.append(width)

    h = int(min(h) / fraction)
    w = int(min(w) / fraction)

    if w % effect_speed == 0:
        k = w // effect_speed
    else:
        k = w // (effect_speed + 1)

    assert duration - k / fps > 0, f"change your parameters, current h = {h}, w = {w}, k = {k}, duration - k / fps = {duration - k / fps}"

    images = list()
    for image in img_list:
        img = cv2.resize(image, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)

    return images, w, h


def cover_animation(img_list, w, h, from_right=random.randint(0, 1), fps=30, effect_speed=2, duration=1):

    frames = []

    if from_right:
        for i in range(len(img_list) - 1):
            j = 0
            for D in range(0, w + 1, effect_speed):
                result = img_list[i].copy()

                result[:, 0:w - D, :] = img_list[i][:, D:w, :]
                result[:, w - D:w, :] = img_list[i + 1][:, 0:D, :]

                frames.append(result)
                j += 1

            # static image in the remaining frames
            for _ in range(fps * duration - j):
                frames.append(img_list[i+1])
    else:
        for i in range(len(img_list) - 1):
            j = 0
            for D in range(0, w + 1, effect_speed):
                result = img_list[i].copy()

                result[:, 0:D, :] = img_list[i + 1][:, w - D:w, :]
                result[:, D:w, :] = img_list[i][:, 0:w - D]

                frames.append(result)
                j += 1

            # static image in the remaining frames
            for _ in range(fps * duration - j):
                frames.append(result)

    return frames


def comb_animation(img_list, w, h, fps=30, effect_speed=2, duration=1):
    lines = random.randint(1, 6)
    frames = []
    h1 = h // lines

    for i in range(len(img_list) - 1):
        j = 0
        for D in range(0, w + 1, effect_speed):
            result = img_list[0].copy()
            for L in range(0, lines, 2):
                result[h1 * L:h1 * (L + 1), 0:D, :] = img_list[i + 1][h1 * L:h1 * (L + 1), w - D:w, :]
                result[h1 * L:h1 * (L + 1), D:w, :] = img_list[i][h1 * L:h1 * (L + 1), 0:w - D]
                result[h1 * (L + 1):h1 * (L + 2), 0:w - D, :] = img_list[i][h1 * (L + 1):h1 * (L + 2), D:w, :]
                result[h1 * (L + 1):h1 * (L + 2), w - D:w, :] = img_list[i + 1][h1 * (L + 1):h1 * (L + 2), 0:D, :]

            frames.append(result)
            j += 1

        # static image in the remaining frames
        for k in range(fps * duration - j):
            frames.append(img_list[i+1])
            
    return frames


def push_animation(img_list, w, h, fps=30, effect_speed=2, duration=1):
    frames = []

    for i in range(len(img_list) - 1):
        j = 0
        for D in range(0, h + 1, effect_speed):
            result = img_list[i].copy()
            result[0:h - D, :, :] = img_list[i][D:h, :, :]
            result[h - D:h, :, :] = img_list[i + 1][0:D, :, :]

            frames.append(result)
            j += 1

        # static image in the remaining frames
        for k in range(fps * duration - j):
            frames.append(img_list[i + 1])

    return frames


def uncover_animation(img_list, w, h, fps=30, effect_speed=2, duration=1):
    frames = []

    for i in range(len(img_list) - 1):
        j = 0
        for D in range(0, w + 1, effect_speed):
            result = img_list[i].copy()
            result[:, 0:w - D, :] = img_list[i][:, D:w, :]
            result[:, w - D:w, :] = img_list[i + 1][:, w - D:w, :]

            frames.append(result)
            j += 1

        # static image in the remaining frames
        for k in range(fps * duration - j):
            frames.append(img_list[i + 1])

    return frames


def split_animation(img_list, w, h, fps=30, effect_speed=2, duration=1):
    frames = []

    for i in range(len(img_list) - 1):
        j = 0
        for D in range(0, w // 2, effect_speed):
            result = img_list[i].copy()
            result[:, w // 2 - D:w // 2 + D, :] = img_list[i + 1][:, w // 2 - D:w // 2 + D, :]
            result[:, 0:w // 2 - D, :] = img_list[i][:, 0:w // 2 - D, :]
            result[:, w // 2 + D:w, :] = img_list[i][:, w // 2 + D:w, :]

            frames.append(result)
            j += 1

        # static image in the remaining frames
        for k in range(fps * duration - j):
            frames.append(img_list[i + 1])

    return frames


def extract_final_H(opencv_bbox, W, H):
    x, y, w, h = convert_bounding_box(box=opencv_bbox, input_type="opencv", change_to="coco")
    
    # All points are in format [cols, rows]
    pt_A, pt_B, pt_C, pt_D = [x, y], [x, y+h], [x+w, y+h], [x+w, y]
    
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                            [0, H - 1],
                            [W - 1, H - 1],
                            [W - 1, 0]])
    
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    
    return M
    
    
def zoom_in_animation(img, W, H, opencv_bbox, fps = 30, duration = 1): 
    
    # x, y, w, h = convert_bounding_box(box=open_cv_bbox, input_type="opencv", change_to="coco")
    final_H = np.matrix.round(extract_final_H(opencv_bbox=opencv_bbox, W=W, H=H), decimals=5, out=None)
    
    frames = []
    
    j=0
    f = duration*fps
    for k in range(0, f+1, 1):
        result = img.copy()
        
        _H = np.array([[1+k*(final_H[0][0]-1)/f, 0, k*(final_H[0][2])/f],
                      [0, 1+k*(final_H[1][1]-1)/f, k*(final_H[1][2])/f],
                      [0, 0, 1]]
                    )
        
        result = cv2.warpPerspective(img, _H, (W, H), flags=cv2.INTER_LINEAR)

        frames.append(result)
        j += 1
                
    return frames
    

def fade_animation(img_list, w, h, fps=30, effect_speed=2, duration=1):
    
    frames = []
    n_frames = int(2*fps*duration/3)
    
    for i in range(len(img_list)-1):
    
        for IN in range(0, n_frames):
        
            fadein = IN/float(n_frames)
            dst = cv2.addWeighted(img_list[i], 1-fadein, img_list[i+1], fadein, 0)
            frames.append(dst)
            
        for _ in range(fps*duration-n_frames):
            frames.append(img_list[i+1])
            
    return frames


def extract_vid(frames, output_path, w=500, h=500, fps=30):
    out = cv2.VideoWriter(r"{}".format(output_path), cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))

    for image in frames:
        for frame in image:
            out.write(frame)

    out.release()
