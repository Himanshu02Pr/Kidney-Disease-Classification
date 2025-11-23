from Classifier.config.configuration import ConfigurationManager
from Classifier.components.base_model import PrepareBaseModel
from Classifier import logger


STAGE_NAME = "Prepare base model"


class BaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        base_model_config = config.get_base_model_config()
        base_model = PrepareBaseModel(config=base_model_config)
        base_model.get_base_model()
        base_model.update_base_model()


    
if __name__ == '__main__':
    try:
        logger.info(f"<-------- {STAGE_NAME} started -------->")
        obj = BaseModelPipeline()
        obj.main()
        logger.info(f"<-------- {STAGE_NAME} completed -------->")
    except Exception as e:
        logger.exception(e)
        raise e