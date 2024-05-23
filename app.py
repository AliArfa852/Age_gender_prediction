# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_age_and_gender
from PIL import Image
from io import BytesIO
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'}), 400
    
    image = request.files['image']

    # Open the image using PIL
    img = Image.open(BytesIO(image.read()))
    img = img.resize((224, 224))  # Resize the image if needed

    predicted_age,predicted_gender = predict_age_and_gender(img)

    # Convert predicted_age to regular Python float
    predicted_age = float(predicted_age)

    return jsonify({"age": predicted_age,
                    "gender":predicted_gender}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)
