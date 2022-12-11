import json
from argparse import ArgumentParser, Namespace
from flask import Flask, abort, request, send_file
import io
import logging
from PIL import Image
import traceback
from timeit import default_timer as time
import cv2

from model_serving import FunctionServingWrapper
from onnx_ritm import RITMInference

logging.basicConfig(level=logging.INFO)


def args_parse() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--ip', required=False, type=str, default='0.0.0.0')
    parser.add_argument('--port', required=False, type=int, default=9019)
    parser.add_argument(
        '--model', required=False, type=str,
        default='./ritm.onnx',
        help='Path to ONNX RITM model'
    )
    return parser.parse_args()


app = Flask(__name__)
app_log = logging.getLogger('werkzeug')
app_log.setLevel(logging.INFO)
ritm_serve: FunctionServingWrapper = None


def serve_pil_image(pil_img):
    img_io = io.BytesIO()
    pil_img.save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpg')


@app.route('/predict', methods=['POST'])
def server_inference():
    logging.info(f'{request.remote_addr}   predict ')
    global ritm_serve

    result: Image = None

    try:
        start_time = time()
        points_data = json.load(request.files['points'])
        # image: Image = Image.open(points_data['image']).convert('RGB')
        image = cv2.cvtColor(cv2.imread(points_data['image'], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image: Image = Image.fromarray(image)
        read_time = time() - start_time
        logging.info('Read time: {:.5f}'.format(read_time))

        start_time = time()
        result = ritm_serve(image, points_data['fg_points'], points_data['bg_points'])
        inference_time = time() - start_time
        logging.info('Inference time: {:.5f}'.format(inference_time))

        start_time = time()
        result = Image.fromarray(result)
        convert_time = time() - start_time
        logging.info('PIL converting time: {:.5f}'.format(convert_time))
    except Exception as e:
        logging.error(
            'server_inference: traced exception'
            '{}: \'{}'.format(
                e, traceback.format_exc()
            )
        )
        abort(502)

    return serve_pil_image(result)



@app.route('/debug_predict', methods=['POST'])
def debug_server_inference():
    logging.info(f'{request.remote_addr}   predict ')
    global ritm_serve

    result: Image = None

    try:
        start_time = time()
        image: Image = Image.open(request.files['image']).convert('RGB')
        points_data = json.load(request.files['points'])
        read_time = time() - start_time
        logging.info('Read time: {:.5f}'.format(read_time))

        start_time = time()
        result = ritm_serve(image, points_data['fg_points'], points_data['bg_points'])
        inference_time = time() - start_time
        logging.info('Inference time: {:.5f}'.format(inference_time))

        start_time = time()
        result = Image.fromarray(result)
        convert_time = time() - start_time
        logging.info('PIL converting time: {:.5f}'.format(convert_time))
    except Exception as e:
        logging.error(
            'server_inference: traced exception'
            '{}: \'{}'.format(
                e, traceback.format_exc()
            )
        )
        abort(502)

    return serve_pil_image(result)


if __name__ == '__main__':
    args = args_parse()
    ritm_serve = FunctionServingWrapper(
        [RITMInference(model_path=args.model)]
    ).start()

    app.run(host=args.ip, debug=False, port=args.port)
