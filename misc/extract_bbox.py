import cv2


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
        x, y = int(bbox[0]), int(bbox[1])  # top-left x, y coordinates
        w, h = int(bbox[2]), int(bbox[3])  # w, h values

        if y - int(p * w) < 0 or x - int(p * h) < 0 or y + int(p * w) > img.shape[0] or y + int(p * w) > img.shape[1] \
                or x + int(p * w) > img.shape[1] or x + int(p * w) > img.shape[0]:
            data.append(img[y:y + w, x:x + h])
        else:
            data.append(img[y - int(p * w):y + w + int(p * w), x - int(p * h):x + h + int(p * h)])  # target box

    return data


def zoom_rescale_bbox(coco_bbox, W, H):
    """
        This function returns a new opencv formatted bounding box output that fits the width-height ratio of the initial image.
        Parameters
        ----------
        + opencv_bbox: list.
            List for the initial bounding box.
        + W: int.
            The width of the image.
        + H: int.
            The height of the image.
        Return
        ----------
            Returns an opencv formatted bounding box.
    """
    
    # input: coco_bbox -> output: opencv_bbox
    x, y, w, h = coco_bbox
    x_top, x_bot, y_top, y_bot = convert_bounding_box(box=coco_bbox, input_type="coco", change_to="opencv")
    if h >= w:
        S = H/h  # scale
        _w = int(W/S)
        _deltaW = _w - w
        if x_top - _deltaW/2 < 0:
            return S, [0, y_top, x_bot+_deltaW-x_top, y_bot]
        elif x_bot + _deltaW/2 > W:
            return S, [x_top + _deltaW - (W-x_bot)-1, y_top, W-1, y_bot]
        else:
            return S, [int(x_top-_deltaW/2), y_top, int(x_bot+_deltaW/2), y_bot]
        
    elif w > h:
        S = W/w
        _h = int(H/S)
        _deltaH = _h - h
        if y_top - _deltaH/2 < 0:
            return S, [x_top, 0, x_bot, y_bot+_deltaH-y_top]
        elif y_top + _deltaH/2 > H:
            return S, [x_top, y_top+_deltaH-(H-y_bot)-1, x_bot, H-1]
        else:
            return S, [x_top, int(y-_deltaH/2), x_bot, int(y+_deltaH/2)]