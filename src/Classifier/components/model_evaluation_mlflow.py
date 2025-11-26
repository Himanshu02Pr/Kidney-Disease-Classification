import tensorflow as tf
from pathlib import Path
import mlflow
import dagshub
from urllib.parse import urlparse
from Classifier.entity.config_entity import EvaluationConfig
from Classifier.utils.common import save_json
import tensorflow as tf


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def load_data(self):
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.data_path,
            labels='inferred',
            label_mode='categorical',
            validation_split=0.2,
            subset='validation',
            image_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            seed=42
        )
        return val_ds

    def evaluation(self):
        val_ds = self.load_data()
        self.model = self.load_model(self.config.path_of_model)
        self.score = self.model.evaluate(val_ds)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    dagshub.init(repo_owner='Himanshu02Pr', repo_name='Kidney-Disease-Classification', mlflow=True)

    def log_into_mlflow(self):     
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                    {"loss": self.score[0], "accuracy": self.score[1]}
                )