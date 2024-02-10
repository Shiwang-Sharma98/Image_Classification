from flask import Flask, request, jsonify, render_template
import os
#from prediction.utils import decodeImage
#from prediction.predict import Dog_or_Cat
from dataclasses import dataclass
from src.image_classification.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.image_classification.pipeline.prepare_base_model import PrepareBaseModelTrainingPipeline
from src.image_classification.pipeline.training_pipeline import ModelTrainingPipeline
from src.image_classification.utils import decodeImage
from src.image_classification.pipeline.predict import PredictionPipeline 




app = Flask(__name__)

@dataclass
class ClientApp:
     filename = "inputImage.jpg"
     classifier = PredictionPipeline(filename)


@app.route("/", methods=['GET'])
def home():
   return render_template('index.html')

@app.route("/train", methods=['POST','GET'])
def trainModel():
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    base_model = PrepareBaseModelTrainingPipeline()
    base_model.main()
    training_model = ModelTrainingPipeline()
    training_model.main()

@app.route("/predict", methods=['POST'])
def predictRoute():
    clApp = ClientApp()
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
