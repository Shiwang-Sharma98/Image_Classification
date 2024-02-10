import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from src.image_classification.entity.config_entity import TrainingConfig
from pathlib import Path

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    def train_valid_generator(self):
        
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
                                   )
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                    rescale = 1./255
                                )
        
        self.training_set = train_datagen.flow_from_directory(self.config.training_data,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
        
        self.test_set = test_datagen.flow_from_directory(self.config.validation_data,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

        
        

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self):
        self.length_of_test_set = len(self.training_set)
        self.length_of_validation_set = len(self.test_set)
        self.steps_per_epoch = self.length_of_test_set
        self.validation_steps = self.length_of_validation_set

        self.model.fit(
            self.training_set,
            validation_data=self.test_set,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )