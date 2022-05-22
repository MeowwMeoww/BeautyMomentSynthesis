import random
import cv2
from tqdm import tqdm
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips


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
        + box: list.
            The provided bounding box in the Python list format. The given bounding box
            must have 4 elements, corresponding to its format.
        + input_type: {'opencv', 'yolo', 'coco'}
            The format of the input bounding box.
            Supported values are 'yolo' - for YOLO format, 'coco' - for COCO format,
            and 'opencv' - for OpenCV format.
        + change_to: {'opencv', 'yolo', 'coco'}.
            The type of format to convert the input bounding box to.
            Supported values are 'yolo' - for YOLO format, 'coco' - for COCO format,
            and 'opencv' - for OpenCV format.
        Return
        ----------
            Returns a list for the converted bounding box.
    """
    assert (type(box) == list), 'The provided bounding box must be a Python list'
    assert (len(box) == 4), 'Must be a bounding box that has 4 elements: [x_left, y_top, x_right, y_bot] (OpenCV format)'
    assert (input_type == 'yolo' or input_type == 'coco' or input_type == 'opencv'), "Must select either 'yolo', 'coco', or 'opencv' as a format of your input bounding box"
    assert (change_to == 'yolo' or change_to == 'coco' or change_to == 'opencv'), "Must select either 'yolo', 'coco', or 'opencv' as a format you want to convert the input bounding box to"
    assert (input_type != change_to), "The format of your input bounding box must be different from your output bounding box."

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
        
        
def get_target_bbox(img, bboxes, p=0.1):
    """
        This function extracts bounding boxes from an image.
        Parameters
        ----------
        + img: numpy array.
            Values of an image (an array of 3 channels).
        + bboxes: list.
            The list of opencv formatted bounding boxes.
        + p: float.
            The coefficient that is used to extend the width and height of the bounding box.
        Return
        ----------
            Returns a list of bounding boxes values in the given image.
    """
    
    data = []
    for bbox in bboxes:
        bbox = convert_bounding_box(box=bbox, input_type="opencv", change_to="coco")
        x, y = int(bbox[0]), int(bbox[1])  # top-left x, y corrdinates
        w, h = int(bbox[2]), int(bbox[3])  # w, h values

        if y - int(p * w) < 0 or x - int(p * h) < 0 or y + int(p * w) > img.shape[0] or y + int(p * w) > img.shape[1] \
                or x + int(p * w) > img.shape[1] or x + int(p * w) > img.shape[0]:
            data.append(img[y:y + w, x:x + h])
        else:
            data.append(img[y - int(p * w):y + w + int(p * w), x - int(p * h):x + h + int(p * h)])  # target box

    return data


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
        k = (w // effect_speed) + 1
        
    assert duration - k / fps > 0, f"change your parameters, current h = {h}, w = {w}, k = {k}, duration - k / fps = {duration - k / fps}"

    images = list()
    for image in img_list:
        img = cv2.resize(image, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    return images, w, h


def cover_animation(img_list, w, h, output_path, from_right=random.randint(0, 1), fps=30, effect_speed=2, duration=1):

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

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
      writer.write(frame)  # write frame into output vid
    writer.release()


def comb_animation(img_list, w, h, output_path, fps=30, effect_speed=2, duration=1):
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
            
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
      writer.write(frame)  # write frame into output vid
    writer.release()


def push_animation(img_list, w, h, output_path, fps=30, effect_speed=2, duration=1):
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

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
      writer.write(frame)  # write frame into output vid
    writer.release()


def uncover_animation(img_list, w, h, output_path, fps=30, effect_speed=2, duration=1):
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

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
      writer.write(frame)  # write frame into output vid
    writer.release()


def split_animation(img_list, w, h, output_path, fps=30, effect_speed=2, duration=1):
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

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
      writer.write(frame)  # write frame into output vid
    writer.release()


def extract_final_H(opencv_bbox, W, H):
    print(opencv_bbox)
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
    
    
def zoom_animation(img, output_path, W, H, opencv_bbox, fps = 30, duration = 1): 
    
    # x, y, w, h = convert_bounding_box(box=open_cv_bbox, input_type="opencv", change_to="coco")
    final_H = np.matrix.round(extract_final_H(opencv_bbox=opencv_bbox, W=W, H=H), decimals=5, out=None)
    
    frames = []
    
    j=0
    f = int(3*(duration*fps+1)/7)
    for k in range(0, f+1, 1):
        result = img.copy()
        
        _H = np.array([[1+k*(final_H[0][0]-1)/f, 0, k*(final_H[0][2])/f],
                      [0, 1+k*(final_H[1][1]-1)/f, k*(final_H[1][2])/f],
                      [0, 0, 1]]
                    )
        
        result = cv2.warpPerspective(img, _H, (W, H), flags=cv2.INTER_LINEAR)

        frames.append(result)
        j += 1
    
    for k in range(f, 1, -1):
        result = img.copy()
        
        _H = np.array([[1+k*(final_H[0][0]-1)/f, 0, k*(final_H[0][2])/f],
                      [0, 1+k*(final_H[1][1]-1)/f, k*(final_H[1][2])/f],
                      [0, 0, 1]]
                    )
        
        result = cv2.warpPerspective(img, _H, (W, H), flags=cv2.INTER_LINEAR)

        frames.append(result)
        j += 1

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for frame in frames:
      writer.write(frame)  # write frame into output vid
    writer.release()
    

def rotate_animation(img, output_path, W, H, opencv_bbox, fps = 30, duration = 1): 
    
    # x, y, w, h = convert_bounding_box(box=open_cv_bbox, input_type="opencv", change_to="coco")
    final_H = np.matrix.round(extract_final_H(opencv_bbox=opencv_bbox, W=W, H=H), decimals=5, out=None)
    
    frames = []
    
    j=0
    f = int(3*(duration*fps+1)/7)
    for k in range(1, f+1, 1):
        result = img.copy()        
        
        x_center, y_center, _, _ = convert_bounding_box(box=opencv_bbox, input_type="opencv", change_to="yolo")

        _R = cv2.getRotationMatrix2D(center=(x_center, y_center), angle=360*k/f, scale=1)
        result = cv2.warpAffine(src=img, M=_R, dsize=(W, H))
        _H = np.array([[(1+k*(final_H[0][0]-1)/f), 0, k*(final_H[0][2])/f],
                      [0,1+k*(final_H[1][1]-1)/f, k*(final_H[1][2])/f],
                      [0, 0, 1]]
                    )
        result = cv2.warpPerspective(result, _H, (W, H), flags=cv2.INTER_LINEAR)
        
        frames.append(result)
        j += 1
    
    for k in range(f, 1, -1):
        result = img.copy()        
        
        x_center, y_center, _, _ = convert_bounding_box(box=opencv_bbox, input_type="opencv", change_to="yolo")

        _R = cv2.getRotationMatrix2D(center=(x_center, y_center), angle=360*k/f, scale=1)
        result = cv2.warpAffine(src=img, M=_R, dsize=(W, H))
        _H = np.array([[(1+k*(final_H[0][0]-1)/f), 0, k*(final_H[0][2])/f],
                      [0,1+k*(final_H[1][1]-1)/f, k*(final_H[1][2])/f],
                      [0, 0, 1]]
                    )
        result = cv2.warpPerspective(result, _H, (W, H), flags=cv2.INTER_LINEAR)
        
        frames.append(result)
        j += 1

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for frame in frames:
      writer.write(frame)  # write frame into output vid
    writer.release()


def fade_animation(img_list, w, h, output_path, fps=30, effect_speed=2, duration=1):
    
    frames = []
    n_frames = int(2*fps*duration/3)
    
    for i in range(len(img_list)-1):
    
        for IN in range(0, n_frames):
        
            fadein = IN/float(n_frames)
            dst = cv2.addWeighted(img_list[i], 1-fadein, img_list[i+1], fadein, 0)
            frames.append(dst)
            
        for _ in range(fps*duration-n_frames):
            frames.append(img_list[i+1])
            
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    for frame in frames:
      writer.write(frame)  # write frame into output vid
    writer.release()


def extract_vid(vid_paths, output_path, w=500, h=500, fps=30):
    print(vid_paths)
    vids = []
    for vid in vid_paths:
        print(vid)
        vids.append(VideoFileClip(vid))

    final = concatenate_videoclips(vids)
    final.write_videofile(output_path, codec="libx264")
