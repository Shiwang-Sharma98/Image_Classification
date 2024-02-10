from src.image_classification.config.configuration import ConfigurationManager
from src.image_classification.components.training import Training 


class ModelTrainingPipeline:
    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()