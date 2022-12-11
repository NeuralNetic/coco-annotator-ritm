from flask_restplus import Namespace, Resource, reqparse
from werkzeug.datastructures import FileStorage
from imantics import Mask
from flask_login import login_required
from config import Config
from PIL import Image
from database import ImageModel
import numpy as np
import io
import json
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import cv2

import os
import logging

logger = logging.getLogger('gunicorn.error')


def get_mask_from_image_by_ritm(
        # _image: Image.Image,
        _image_path: str,
        _points,
        _api: str = 'http://ritmserver:9019/predict') -> np.ndarray:
    # image_file = io.BytesIO()
    # _image.save(image_file, 'PNG')
    # image_file.seek(0)

    mp_encoder = MultipartEncoder(
        fields={
            # 'image': (
            #     'image', image_file,
            #     'image/png'
            # ),
            'points': (
                'points',
                json.dumps({'fg_points': _points[0], 'bg_points': _points[1], 'image': _image_path}),
                'dictionary/json'
            )
        }
    )
    response = requests.post(
        url=_api,
        data=mp_encoder,
        headers={'Content-Type': mp_encoder.content_type}
    )
    response_status_code = response.status_code

    if response_status_code != 200:
        raise RuntimeError(
            'Status mask server: {}'.format(
                response.status_code)
        )

    mask = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    response.close()

    return mask


MASKRCNN_LOADED = os.path.isfile(Config.MASK_RCNN_FILE)
if MASKRCNN_LOADED:
    from ..util.mask_rcnn import model as maskrcnn
else:
    logger.warning("MaskRCNN model is disabled.")

DEXTR_LOADED = os.path.isfile(Config.DEXTR_FILE)
if DEXTR_LOADED:
    from ..util.dextr import model as dextr
else:
    logger.warning("DEXTR model is disabled.")


RITM_LOADED = os.path.isfile(Config.RITM_FILE)
if not RITM_LOADED:
    logger.warning("RITM model is disabled.")

api = Namespace('model', description='Model related operations')


image_upload = reqparse.RequestParser()
image_upload.add_argument('image', location='files', type=FileStorage, required=True, help='Image')

dextr_args = reqparse.RequestParser()
dextr_args.add_argument('points', location='json', type=list, required=True)
dextr_args.add_argument('padding', location='json', type=int, default=50)
dextr_args.add_argument('threshold', location='json', type=int, default=80)


@api.route('/ritm/<int:image_id>')
class MaskRCNN(Resource):

    @login_required
    @api.expect(dextr_args)
    def post(self, image_id):
        """ COCO data test """

        if not RITM_LOADED:
            return {"disabled": True, "message": "RITM is disabled"}, 400

        args = dextr_args.parse_args()
        # fg_points = args.get('fg_points')
        # bg_points = args.get('bg_points')
        points =  args.get('points')
        # padding = args.get('padding')
        # threshold = args.get('threshold')

        if len(points) == 0:
            return {"message": "Invalid points entered"}, 400

        image_model = ImageModel.objects(id=image_id).first()
        if not image_model:
            return {"message": "Invalid image ID"}, 400

        fg_points = [p[:2] for p in points if p[2] == 1]
        bg_points = [p[:2] for p in points if p[2] == 0]

        # image = Image.open(image_model.path)
        result = get_mask_from_image_by_ritm(str(image_model.path), [fg_points, bg_points])

        return {"segmentaiton": Mask(result).polygons().segmentation}


@api.route('/dextr/<int:image_id>')
class MaskRCNN(Resource):

    @login_required
    @api.expect(dextr_args)
    def post(self, image_id):
        """ COCO data test """

        if not DEXTR_LOADED:
            return {"disabled": True, "message": "DEXTR is disabled"}, 400

        args = dextr_args.parse_args()
        points = args.get('points')
        # padding = args.get('padding')
        # threshold = args.get('threshold')

        if len(points) != 4:
            return {"message": "Invalid points entered"}, 400
        
        image_model = ImageModel.objects(id=image_id).first()
        if not image_model:
            return {"message": "Invalid image ID"}, 400
        
        image = Image.open(image_model.path)
        result = dextr.predict_mask(image, points)

        return { "segmentaiton": Mask(result).polygons().segmentation }


@api.route('/maskrcnn')
class MaskRCNN(Resource):

    @login_required
    @api.expect(image_upload)
    def post(self):
        """ COCO data test """
        if not MASKRCNN_LOADED:
            return {"disabled": True, "coco": {}}

        args = image_upload.parse_args()
        im = Image.open(args.get('image'))
        coco = maskrcnn.detect(im)
        return {"coco": coco}