import random
import cv2 
import collections
import numpy as np

def random_mirror(imgs):
    outputs = {}
    if random.random() > 0.5:
        for key, img in imgs.items():
            if img is not None:
                outputs[key] = cv2.flip(img, 1)
            else:
                outputs[key] = None
    else:
        outputs = imgs
    return outputs

#DOESN'T HANDLE GT CUTOUT (VALUE SHOULD BE 255 FOR GT)
# def cutout(imgs, imgsize, keys_to_cutout, cutoutsize = 50):
#     outputs = {}
#     h0 = random.randrange(imgsize[0] - cutoutsize)
#     w0 = random.randrange(imgsize[1] - cutoutsize)
#     for key, img in imgs.items():
#         if key in keys_to_cutout:
#             if img is not None:
#                 avg = np.mean(img, axis = (0, 1))
#                 img[h0:h0+cutoutsize, w0:w0 + cutoutsize] = avg
#                 outputs[key] = img
#             else:
#                 outputs[key] = None
#     return outputs


def random_scale(imgs, scale_array, orig_size):
    scale = random.choice(scale_array)
    sh = int(orig_size[0] * scale)
    sw = int(orig_size[1] * scale)
    outputs = {}
    for key, img in imgs.items():
        if img is not None:
            if key == 'rgb':
                outputs[key] = resizergb(img, (sw, sh))
            elif key == 'depth':
                outputs[key] = resizedepth(img, (sw, sh))
            elif key == 'gt':
                outputs[key] = resizegt(img, (sw, sh))
            else: 
                raise Exception(key, "not supported in random_scale")
        else:
            outputs[key] = None
    return outputs, scale


def get_2dshape(shape, *, zero=True):
    if not isinstance(shape, collections.Iterable):
        shape = int(shape)
        shape = (shape, shape)
    else:
        h, w = map(int, shape)
        shape = (h, w)
    if zero:
        minv = 0
    else:
        minv = 1

    assert min(shape) >= minv, 'invalid shape: {}'.format(shape)
    return shape


def generate_random_crop_pos(ori_size, crop_size):
    ori_size = get_2dshape(ori_size)
    h, w = ori_size

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    pos_h, pos_w = 0, 0

    if h > crop_h:
        pos_h = random.randint(0, h - crop_h + 1)

    if w > crop_w:
        pos_w = random.randint(0, w - crop_w + 1)

    return pos_h, pos_w


def pad_image_to_shape(img, shape, border_mode, value):
    # print("enter pad image", img.shape, np.mean(img[:, :, 3]), np.max(img[:, :, 3]), np.mean(img[:, :, 0]), np.max(img[:, :, 0]))
    margin = np.zeros(4, np.uint32)
    shape = get_2dshape(shape)
    pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
    pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

    margin[0] = pad_height // 2
    margin[1] = pad_height // 2 + pad_height % 2
    margin[2] = pad_width // 2
    margin[3] = pad_width // 2 + pad_width % 2

    img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3],
                             border_mode, value=value)

    return img, margin


def random_crop_pad_to_shape(imgs, img_size, crop_size):
    crop_pos = generate_random_crop_pos(img_size, crop_size)
    h, w = img_size
    start_crop_h, start_crop_w = crop_pos
    assert ((start_crop_h < h) and (start_crop_h >= 0))
    assert ((start_crop_w < w) and (start_crop_w >= 0))

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    outputs = {}
    for key, img in imgs.items():
        if img is not None:
            img_crop = img[start_crop_h:start_crop_h + crop_h,
                        start_crop_w:start_crop_w + crop_w, ...]
            if key == 'rgb':
                pad_label_value = 0
            elif key == 'depth':
                pad_label_value = 0
            elif key == 'gt':
                pad_label_value = 255
            else:
                raise Exception(f"pad_label_value not defined for {key} in random_crop_pad_to_shape")
            
            img, margin = pad_image_to_shape(img_crop, crop_size, cv2.BORDER_CONSTANT,
                                            pad_label_value)
            outputs[key] = img 
        else:
            outputs[key] = None        
    return outputs, margin


def normalize(img, mean, std):
    # pytorch pretrained model need the input range: 0-1
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img / std

    return img

def normalizedepth(img):
    # pytorch pretrained model need the input range: 0-1
    img = img.astype(np.float32) / 255.0
    return img

def tfnyu_normalizedepth(img):
    # pytorch pretrained model need the input range: 0-1
    img = img.astype(np.float32) / 5000.
    return img


def resizergb(rgb, expectedshape):
    return cv2.resize(rgb, expectedshape, interpolation=cv2.INTER_LINEAR)

def resizedepth(depth, expectedshape):
    return cv2.resize(depth, expectedshape, interpolation=cv2.INTER_NEAREST)    

def resizegt(gt, expectedshape):
    return cv2.resize(gt, expectedshape, interpolation=cv2.INTER_NEAREST)