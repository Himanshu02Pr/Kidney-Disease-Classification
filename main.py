import os
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU is configured with memory growth enabled.")
    except RuntimeError as e:
        print(e)

from Classifier import logger
from Classifier.pipeline.Stage01_data_ingestion import DataIngestionPipeline
from Classifier.pipeline.Stage02_base_model import BaseModelPipeline
from Classifier.pipeline.Stage03_model_training import ModelTrainingPipeline
from Classifier.pipeline.Stage04_model_evaluation import EvaluationPipeline



STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f"\n\n<-------- {STAGE_NAME} started -------->\n\n") 
   data_ingestion = DataIngestionPipeline()
   data_ingestion.main()
   logger.info(f"\n\n<-------- {STAGE_NAME} completed -------->\n\n")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"\n\n<-------- {STAGE_NAME} started -------->\n\n") 
   prepare_base_model = BaseModelPipeline()
   prepare_base_model.main()
   logger.info(f"\n\n<-------- {STAGE_NAME} completed -------->\n\n")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Training"
try: 
   logger.info(f"\n\n<-------- {STAGE_NAME} started -------->\n\n")
   model_trainer = ModelTrainingPipeline()
   model_trainer.main()
   logger.info(f"\n\n<-------- {STAGE_NAME} completed -------->\n\n")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Evaluation stage"
try:
   logger.info(f"\n\n<-------- {STAGE_NAME} started -------->\n\n")
   model_evalution = EvaluationPipeline()
   model_evalution.main()
   logger.info(f"\n\n<-------- {STAGE_NAME} completed -------->\n\n")

except Exception as e:
        logger.exception(e)
        raise e