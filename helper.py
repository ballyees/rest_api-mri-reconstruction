import tensorflow as tf
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np

from os.path import join, dirname, abspath

import cv2
def load_keras_model(path):
    return load_model(join(dirname(abspath(__file__)), path))

def read_image_base64(image):
    image = base64.b64decode(image)
    image = Image.open(BytesIO(image)).convert('RGB')
    image = np.asarray(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    image = np.expand_dims(image, 0)
    return image / 255

def round_image(image_norm):
    image_norm = np.round(image_norm * 255)
    return image_norm.astype(int)

def image_to_json(image):
    image = np.squeeze(image)
    image = round_image(image)
    _, img_bytes = cv2.imencode('.png', image)
    img_bytes = base64.b64encode(img_bytes)
    return img_bytes.decode()