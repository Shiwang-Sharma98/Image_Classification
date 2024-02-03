from flask import Flask, request, jsonify, render_template
import os
from prediction.utils import decodeImage
from prediction.predict import Dog_or_Cat
from dataclasses import dataclass


app = Flask(__name__)

@dataclass
class ClientApp:
    filename = "inputImage.jpg"
    classifier = Dog_or_Cat(filename)


@app.route("/", methods=['GET'])

def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predictRoute():
    clApp = ClientApp()
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict_dogcat()
    return jsonify(result)




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
