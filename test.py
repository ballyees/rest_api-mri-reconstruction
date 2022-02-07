import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

def image_decode(image):
    image = base64.b64decode(image)
    image = Image.open(BytesIO(image)).convert('RGB')
    image = np.asarray(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    return image
with open('t.png', 'rb') as file:
    # ba = bytearray(file.read())
    ba = Image.open(file).convert('RGB')
    print(ba)
    img = np.asarray(ba, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(img.shape)
    print('-----------'*10)
    _, img_bytes = cv2.imencode('.png', img)
    # print(img_bytes1)
    img_bytes = base64.b64encode(img_bytes)
    img_bytes = img_bytes.decode()
    image = base64.b64decode(img_bytes)
    image = Image.open(BytesIO(image)).convert('RGB')
    image = np.asarray(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    print(image.dtype, img.dtype)
    print((image == img).all())
import requests
data = img_bytes
res = requests.post('http://127.0.0.1:8000/v1/fast/3x', json={'file': data})
img_res = image_decode(res.json()['response'])
cv2.imwrite('res.png', img_res)
# print(img)