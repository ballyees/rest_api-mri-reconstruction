from sanic import Blueprint
from sanic.response import json
import helper
from .models import models

bp_slow = Blueprint('predict-with-64-filters', url_prefix='/slow', version="v1")

@bp_slow.route('/3x', methods=['POST'])
async def pred_64_filter_3x(request):
    json_body = request.json
    img = helper.read_image_base64(json_body['file'])
    pred = models.model_64_3(img)
    pred = helper.image_to_json(pred)
    return json({'response': pred})

@bp_slow.route('/5x', methods=['POST'])
async def pred_64_filter_5x(request):
    json_body = request.json
    img = helper.read_image_base64(json_body['file'])
    pred = models.model_64_5(img)
    pred = helper.image_to_json(pred)
    return json({'response': pred})

@bp_slow.route('/8x', methods=['POST'])
async def pred_64_filter_8x(request):
    json_body = request.json
    img = helper.read_image_base64(json_body['file'])
    pred = models.model_64_8(img)
    pred = helper.image_to_json(pred)
    return json({'response': pred})