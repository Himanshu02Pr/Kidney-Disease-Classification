from Classifier import logger
from Classifier.pipeline.Stage01_data_ingestion import DataIngestionPipeline
from Classifier.pipeline.Stage02_base_model import BaseModelPipeline
from Classifier.pipeline.Stage03_model_training import ModelTrainingPipeline



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
