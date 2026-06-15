from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from predict import predict
import os

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../frontend', path)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "DrishtiAI backend is running!"})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_file = request.files['image']
    image_bytes = image_file.read()
    result = predict(image_bytes)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)