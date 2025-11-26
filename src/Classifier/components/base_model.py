import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, models, optimizers, losses
from pathlib import Path
from Classifier.entity.config_entity import BaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: BaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    

    @staticmethod
    def prepare_full_model(image_size, base_model, dropout_rate, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in base_model.layers:
                base_model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in base_model.layers[:-freeze_till]:
                base_model.trainable = False

        full_model = models.Sequential([
            layers.Lambda(
                lambda x: preprocess_input(x), 
                name='vgg16_preprocessing_layer',
                input_shape=image_size
            ),
            base_model,
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(classes, activation='softmax')
        ])

        full_model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=losses.categorical_crossentropy,
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    
    
    def update_base_model(self):
        self.full_model = self.prepare_full_model(
            image_size=self.config.params_image_size,
            base_model=self.model,
            dropout_rate=self.config.params_dropout_rate,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)