from sanic import Blueprint
from sanic.response import json
import helper 
from .models import models
import numpy as np
bp_fast = Blueprint('predict-with-16-filters', url_prefix='/fast', version="v1")

@bp_fast.route('/3x', methods=['POST'])
async def pred_16_filter_3x(request):
    json_body = request.json
    img = helper.read_image_base64(json_body['file'])
    pred = models.model_16_3(img)
    pred = helper.image_to_json(pred)
    return json({'response': pred})

@bp_fast.route('/5x', methods=['POST'])
async def pred_16_filter_5x(request):
    json_body = request.json
    img = helper.read_image_base64(json_body['file'])
    pred = models.model_16_5(img)
    pred = helper.image_to_json(pred)
    return json({'response': pred})

@bp_fast.route('/8x', methods=['POST'])
async def pred_16_filter_8x(request):
    json_body = request.json
    img = helper.read_image_base64(json_body['file'])
    pred = models.model_16_8(img)
    pred = helper.image_to_json(pred)
    return json({'response': pred})