from Classifier import logger
from Classifier.pipeline.Stage01_data_ingestion import DataIngestionPipeline
from Classifier.pipeline.Stage02_base_model import BaseModelPipeline



STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f"<-------- {STAGE_NAME} started -------->") 
   data_ingestion = DataIngestionPipeline()
   data_ingestion.main()
   logger.info(f"<-------- {STAGE_NAME} completed -------->")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"<-------- {STAGE_NAME} started -------->") 
   prepare_base_model = BaseModelPipeline()
   prepare_base_model.main()
   logger.info(f"<-------- {STAGE_NAME} completed -------->")
except Exception as e:
        logger.exception(e)
        raise e