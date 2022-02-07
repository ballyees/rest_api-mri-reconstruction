from sanic import Blueprint
from sanic.response import json
from ..helper import *
from .models import models

bp_fast = Blueprint('predict-with-16-filters', url_prefix='/fast', version="v1")

@bp_fast.route('/3x', methods=['POST'])
async def pred_16_filter_3x(request):
    json_body = request.json
    img = read_image_base64(json_body['file'])
    pred = models.model_16_3(img)
    pred = image_to_json(pred)
    return json({'response': pred})

@bp_fast.route('/5x', methods=['POST'])
async def pred_16_filter_5x(request):
    json_body = request.json
    img = read_image_base64(json_body['file'])
    pred = models.model_16_5(img)
    pred = image_to_json(pred)
    return json({'response': pred})

@bp_fast.route('/8x', methods=['POST'])
async def pred_16_filter_8x(request):
    json_body = request.json
    img = read_image_base64(json_body['file'])
    pred = models.model_16_8(img)
    pred = image_to_json(pred)
    return json({'response': pred})