from cv2 import magnitude
import numpy as np
import tensorflow as tf
import fastmri
import cv2
import h5py
from sklearn.metrics import normalized_mutual_info_score
from os import mkdir
from fastmri.data import transforms as T
try:
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

def read_kspace_multiple_file(file_names, target_shape=None):
    data = []
    for fname in file_names:
        with h5py.File(fname, 'r') as hdf:
            if target_shape is None:
                data.append(hdf['kspace'][()])
            else:
                kspace, start, end = center_crop(hdf['kspace'][()], shape=target_shape)
                data.append(kspace)
    return np.concatenate(data, axis=0)

def norm_minmax(data):
    _min = data.min()
    return (data - _min) / (data.max() - _min)

def norm_center_value(data):
    _max = data.max()
    _min = data.min()
    center_val = (_max + _min) / 2
    return (data - center_val) / (_max - center_val), (_min, _max, center_val)
def safe_mkdir(path):
    try:
        mkdir(path)
    except FileExistsError as err:
        print(err)
def norm_init_center_value(data, _min, _max, center_val):
    return (data - center_val) / (_max - center_val)
def inverse_norm_center_value(data, _min, _max, center_val):
    return (data * (_max - center_val)) + center_val
def find_images_max(images):
    return np.array([tf.reduce_max(ci) for ci in images])
def find_images_min(images):
    return np.array([tf.reduce_min(ci) for ci in images])
def norm_images_center_value(images, keep_inverse=False):
    _min = find_images_min(images)
    _max = find_images_max(images)
    _min = np.expand_dims(_min, axis=tuple(i for i in range(1, len(images.shape))))
    _max = np.expand_dims(_max, axis=tuple(i for i in range(1, len(images.shape))))
    center = (_max - _min) / 2
    if keep_inverse:
        return (images-center) / (_max-center), (_min.flatten(), _max.flatten(), center.flatten())
    else:
        return (images-center) / (_max-center)
def inverse_norm_images_center_value(images, _min, _max, _center):
    max_expand = np.expand_dims(_max, axis=tuple(i for i in range(1, len(images.shape))))
    center_expand = np.expand_dims(_center, axis=tuple(i for i in range(1, len(images.shape))))
    return (images * (max_expand - center_expand)) + center_expand

def norm_kspace(kspace, keep_inverse=False, use_coil=True):
    magnitude = np.abs(kspace)
    if use_coil is True:
        _max = np.max(magnitude, axis=(-2, -1)) # image
        _min = np.min(magnitude, axis=(-2, -1))
        _max = np.expand_dims(_max, axis=tuple(i for i in range(2, kspace.ndim)))
        _min = np.expand_dims(_min, axis=tuple(i for i in range(2, kspace.ndim)))
    else:
        _max = np.max(magnitude, axis=(-3, -2, -1)) # coil, image
        _min = np.min(magnitude, axis=(-3, -2, -1))
        _max = np.expand_dims(_max, axis=tuple(i for i in range(1, kspace.ndim)))
        _min = np.expand_dims(_min, axis=tuple(i for i in range(1, kspace.ndim)))
    if keep_inverse:
        return (kspace-_min) / (_max-_min), (_min.flatten(), _max.flatten())
    else:
        return (kspace-_min) / (_max-_min)
def inverse_norm_kspace(kspace, _min, _max, use_coil=True):
    if use_coil is True:
        max_expand = np.expand_dims(_max, axis=tuple(i for i in range(1, kspace.ndim)))
        min_expand = np.expand_dims(_min, axis=tuple(i for i in range(1, kspace.ndim)))
    else:
        max_expand = np.expand_dims(_max, axis=tuple(i for i in range(2, kspace.ndim)))
        min_expand = np.expand_dims(_min, axis=tuple(i for i in range(2, kspace.ndim)))
    return (kspace * (max_expand - min_expand)) + min_expand
def center_crop(data, shape):
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to], (w_from, h_from), (w_to, h_to)

def get_center_slice(data, num_slice=32):
    c_num_slice = int(num_slice//2)
    if num_slice > data.shape[0]:
        raise ValueError("Invalid num_slice.")
    c_slice = int(data.shape[0]//2)
    start = c_slice-c_num_slice
    return data[start:start+num_slice]

def create_mask(shape, speedup=8, center_size=50, thickness=2):
    assert len(shape) == 2
    mask = np.zeros(shape[1])
    s = 100/speedup
    size = int(shape[1]*s/100)
    center = int(shape[1]//2)
    c_size = int((size * center_size) / 100)
    left_size = size - c_size
    rand_size = int((left_size / 2) / thickness)
    start_idx = center-int(c_size/2)
    mask[start_idx:start_idx+c_size] = 1
    l = np.arange(0, start_idx, thickness)
    r = np.arange(start_idx+c_size, shape[1], thickness)
    np.random.shuffle(l)
    np.random.shuffle(r)
    for i in range(rand_size):
        mask[r[i]:r[i]+thickness] = 1
        mask[l[i]:l[i]+thickness] = 1
    return np.tile(mask, (shape[0], 1)).astype(int)

def create_mask_uniform(shape, space=11, center_size=8, thickness=2):
    assert len(shape) == 2
    mask = np.zeros(shape[1])
    center = int(shape[1]//2)
    c_size = int((shape[1]*center_size)/100)
    start_idx = center-int(c_size/2)
    mask[start_idx:start_idx+c_size] = 1
    for i in range(start_idx-space, 0, -space):
        mask[i:i+thickness] = 1
    for i in range(start_idx+c_size, shape[1], space):
        mask[i:i+thickness] = 1
    return np.tile(mask, (shape[0], 1)).astype(int)

def create_mask_speed_up(target_shape, speed_up):
    if speed_up == '2x':
        mask = create_mask_uniform(target_shape, space=3, thickness=1, center_size=25) # ~2x+
    elif speed_up == '2.5x':
        mask = create_mask_uniform(target_shape, space=6, thickness=1, center_size=20) # ~2.5x+
    elif speed_up == '3x':
        mask = create_mask_uniform(target_shape, space=5, thickness=1, center_size=16) # ~3x+
    elif speed_up == '4x':
        mask = create_mask_uniform(target_shape, space=12, thickness=2, center_size=11) # ~4x-
    elif speed_up == '5x':
        mask = create_mask_uniform(target_shape, space=10, center_size=10, thickness=1) # ~5x+
    elif speed_up == '8x':
        mask = create_mask_uniform(target_shape, space=11, center_size=4, thickness=1) # ~8x+
    else:
        raise ValueError('speed up support: [2x, 2.5x, 3x, 4x, 5x, 8x]')
    return mask

def create_mask_speed_up_8(target_shape, speed_up):
    if speed_up == '2x':
        mask = create_mask_uniform(target_shape, space=7, thickness=3, center_size=8) # ~2.11x+
    elif speed_up == '2.6x':
        mask = create_mask_uniform(target_shape, space=6, thickness=2, center_size=8) # ~2.6x+
    elif speed_up == '3x':
        mask = create_mask_uniform(target_shape, space=7, thickness=2, center_size=8) # ~2.9x+
    elif speed_up == '4x':
        mask = create_mask_uniform(target_shape, space=11, thickness=2, center_size=8) # ~4x+
        mask = create_mask_uniform(target_shape, space=6, thickness=1, center_size=8) # ~4.3x+
    elif speed_up == '5x':
        mask = create_mask_uniform(target_shape, space=8, thickness=1, center_size=8) # ~5.1x+
    elif speed_up == '8x':
        mask = create_mask_uniform(target_shape, space=11, thickness=1, center_size=4) # ~8.2x+
    else:
        raise ValueError('speed up support: [2x, 2.5x, 3x, 4x, 5x, 8x]')
    return mask

def crop_mask(mask, start_idx, shape):
    assert len(mask.shape) == 2
    return mask[start_idx:start_idx+shape[0], :shape[1]]

def kspace_to_images(kspace, target_shape=None, rss=True, axis=1):
    if target_shape is not None:
        kspace, start, end = center_crop(kspace, shape=target_shape)
    images = np.fft.ifftshift(kspace, axes=(-2, -1))
    images = np.fft.ifftn(images, axes=(-2, -1))
    images = np.fft.fftshift(images, axes=(-2, -1))
    if rss is True:
        images = root_sum_squared(images, axis=axis)
    return images

def images_to_kspace(images):
    kspace = np.fft.ifftshift(images, axes=(-2, -1))
    kspace = np.fft.fftn(kspace, axes=(-2, -1))
    kspace = np.fft.fftshift(kspace, axes=(-2, -1))
    return kspace

def kspace_to_images_tf(kspace, target_shape=None, rss=True, axis=1):
    if target_shape is not None:
        kspace, start, end = center_crop(kspace, shape=target_shape)
    images = tf.signal.ifftshift(kspace, axes=(-2, -1))
    images = tf.signal.ifft2d(images)
    images = tf.signal.fftshift(images, axes=(-2, -1))
    if rss is True:
        images = root_sum_squared_tf(images, axis=axis)
    return images

def images_to_kspace_tf(images):
    kspace = tf.signal.ifftshift(images, axes=(-2, -1))
    kspace = tf.signal.fft2d(kspace)
    kspace = tf.signal.fftshift(kspace, axes=(-2, -1))
    return kspace

def root_sum_squared(ifft, axis=1):
    images = np.abs(ifft)
    images = np.sum(images, axis=axis)
    return images

def root_sum_squared_tf(ifft, axis=1):
    images = tf.abs(ifft)
    images = tf.reduce_sum(images, axis=axis)
    return images

def apply_mask(img, mask, padding=0):
    mask_size = mask.shape[:2]
    mask_res = np.zeros_like(mask, np.uint8)
    for i in range(mask_size[0]):
        if np.sum(mask[i]) > 0:
            idx = np.argwhere(mask[i]==True).flatten()
            if idx.shape[0] < 10:
                continue
            start = max(0, idx[0] - padding)
            end = min(idx[-1] + padding, mask_size[1])
            idx = np.arange(start, end+1, dtype=np.int16)
            mask_res[i][idx] = 1
    return img * mask_res, mask_res

def coil_sensitivities(kspace, dim=1, threshold=0.12, return_process=True):
    kspace_tensor = T.to_tensor(kspace)
    coil_images = fastmri.complex_abs(fastmri.ifft2c(kspace_tensor))
    images = fastmri.rss(coil_images, dim=dim)
    sensitivities = coil_images / tf.expand_dims(images, 1)
    threshold = find_images_max(images) * threshold
    mask = [np.expand_dims(images[i] > threshold[i], 0) for i in range(threshold.shape[0])]
    mask = np.expand_dims(np.concatenate(mask, axis=0), 1)
    sensitivities = sensitivities * mask
    if return_process:
        return sensitivities, images, coil_images
    else:
        return sensitivities

def remove_bg_kmeans(image, k):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    image_val = image.reshape(-1)
    image_val = np.float32(image_val)
    _, labels, (centers) = cv2.kmeans(image_val, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    labels = labels != labels[0]
    labels = labels.reshape(image.shape)
    segmented_image = (image * labels).astype(np.uint8)
    segmented_image = cv2.medianBlur(segmented_image, 3)
    segmented_image, mask = apply_mask(image, segmented_image > 0, padding=5)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((40, 10), np.uint8))
    return image, mask

def accuracy_score(y_true, y_pred, thresh=0.5):
    y_true = np.reshape(y_true, -1)
    y_pred = np.reshape(y_pred, -1)
    mask = y_pred <= thresh
    y_pred = np.ones_like(y_pred, dtype=np.uint8)
    y_pred[mask] = 0
    # y_pred[~mask] = 1
    return np.sum(y_true == y_pred) / len(y_true)