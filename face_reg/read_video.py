import os

import cv2
import numpy as np
from tqdm.notebook import tqdm


def read_video(path, seconds_per_frame=1, ratio_to_resize=1, read_all=False):
    """
    This function will read an MP4 video file, and convert it to a Numpy Array
    formatted batch of frames.
    To avoid your computer's RAM from becoming overloaded, this function allows
    you to convert frames only after a specific time interval.
    Example:
    Convert the 1st frame --> skip 5 seconds of video --> convert the 2nd frame
    --> ... --> until all frames have been either skipped or converted.

    Parameters
    ----------
    + path : str .
        The provided path to a valid MP4 video file.

    + seconds_per_frame : integer or float > 0, optional, default is 1 .
        The number of seconds in the video are skipped in-between converting
        frames.

    + ratio_to_resize : integer or float > 0, optional, default is 1 .
        The ratio to resize the width and height of the frame.
        Example: If ratio_to_resize is 2, the width and height of the frame will
        be half of their original sizes.

    + read_all: bool, optional, default is False .
        When True, read and convert all the frames of the video.
        When False, only read selected frames after certain time intervals.

    Return
    ----------
    buf : np.ndarray
        Contains all the selected frames in the Python Numpy Array format.

    """

    # Assert all the provided values are of correct types, and are valid.
    assert (type(seconds_per_frame) == int or type(
        seconds_per_frame) == float), "The provided seconds_per_frame value must be an number larger than 0"
    assert (type(ratio_to_resize) == int), "The provided ratio_to_resize value must be an integer larger than 0"
    assert (type(read_all) == bool), "The provided read_all value must be a boolean value"
    assert (os.path.exists(path)), "The provided path must contain an URL of a MP4 file"

    # Create all the necessary variables
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_FPS, 30)  # set fps value of all videos to 30 fps
    fps = 30
    frame_interval = int(
        seconds_per_frame * fps)  # the amount of frames we will skip in-between converting chosen frames

    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    listCount = int(frameCount / frame_interval)
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / ratio_to_resize)  # a frame's width after resize
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / ratio_to_resize)  # a frame's height after resize

    # Read the given MP4 video file and return the converted frames.
    if not read_all:
        print('Total number of frames available in the video:', frameCount)
        print('Total number of frames that we will convert:', listCount)
        frame_count = 0
        list_count = 0
        ret = True

        buf = np.empty((listCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
        pbar = tqdm(total=listCount)

        while list_count < listCount and ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, img = cap.read()
            img = cv2.resize(img, (frameWidth, frameHeight), interpolation=cv2.INTER_AREA)
            buf[list_count] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_count += frame_interval
            list_count += 1
            pbar.update(1)

        pbar.close()

    else:
        print('Total number of frames that we will convert:', frameCount)

        frame_count = 0
        ret = True

        buf = np.empty((frame_count, frameHeight, frameWidth, 3), np.dtype('uint8'))
        pbar = tqdm(total=frameCount)

        while frame_count < frameCount and ret:
            ret, img = cap.read()
            img = cv2.resize(img, (frameWidth, frameHeight), interpolation=cv2.INTER_AREA)
            buf[frame_count] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_count += 1
            pbar.update(1)

        pbar.close()

    return buf
