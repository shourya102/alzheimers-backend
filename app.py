import os
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from img_prediction import predict, preprocessing, get_response_image

app = Flask(__name__)
CORS(app)


@app.route('/image_response', methods=['POST', 'GET'])
def image_response():
    model_name = request.form.get('model_name')
    img = request.files['img']
    img_path = ''
    if img:
        filename = secure_filename(img.filename)
        tempdir = tempfile.gettempdir()
        img_path = os.path.join(tempdir, filename)
        img.save(img_path)
    res = predict(img_path, f'model/{model_name}_alz.h5')
    response = {}
    if res == 'Very Mild Demented':
        response['name'] = res
        response['details'] = 'Slight memory lapses, minimal impact.'
    elif res == 'Mild Demented':
        response['name'] = res
        response['details'] = 'Noticeable memory loss, some daily challenges.'
    elif res == 'Non Demented':
        response['name'] = res
        response['details'] = 'Normal cognitive function.'
    else:
        response['name'] = res
        response['details'] = 'Significant memory impairment, difficulty with tasks.'
    return jsonify(response)


@app.route('/preprocessed_response', methods=['POST'])
def preprocessed_response():
    img = request.files['img']
    img_path = ''
    if img:
        filename = secure_filename(img.filename)
        tempdir = tempfile.gettempdir()
        img_path = os.path.join(tempdir, filename)
        img.save(img_path)
    preprocessed_img = preprocessing(img_path)
    results = [get_response_image(x) for x in preprocessed_img]
    return jsonify({'results': results})


if __name__ == '__main__':
    app.run(debug=True)
