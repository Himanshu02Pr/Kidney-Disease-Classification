from glob import glob
import math
import os
import tensorflow as tf
from pathlib import Path
from Classifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    
    def load_data(self):

        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.training_data,
            labels='inferred',
            label_mode='categorical',
            validation_split=0.2,
            subset='training',
            image_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            seed=42
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.training_data,
            labels='inferred',
            label_mode='categorical',
            validation_split=0.2,
            subset='validation',
            image_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            seed=42
        )

        return train_ds, val_ds
    
    def sample_count(self):
        total_count = 0
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            total_count += len(glob(os.path.join(self.config.training_data, '*', f'*.{ext}')))
        return total_count

    

    def image_aug(self):

        augment_pipeline = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ])
        return augment_pipeline
      
    def train(self):
        train_samples = math.floor(self.sample_count()*0.8)
        val_samples = self.sample_count()-train_samples
        self.steps_per_epoch = train_samples // self.config.params_batch_size
        self.validation_steps = val_samples // self.config.params_batch_size

        AUTOTUNE = tf.data.AUTOTUNE
        augment = self.image_aug()

        train_ds, val_ds = self.load_data()

        if self.config.params_is_augmentation:
            train_ds = train_ds.map(
                lambda x, y : (augment(x, training=True),y),
                num_parallel_calls = AUTOTUNE
            )
        else:
            train_ds

        train_data = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_data = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


        self.model.fit(
            train_data,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=val_data
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
