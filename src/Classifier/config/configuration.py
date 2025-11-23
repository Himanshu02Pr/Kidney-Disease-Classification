from Classifier.constants import *
from Classifier.utils.common import read_yaml, create_directories
from Classifier.entity.config_entity import DataIngestionConfig, BaseModelConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        
        data_ingestion_config = DataIngestionConfig(
            source=config.kaggle_source,
            download_path=config.download_path,
        )

        return data_ingestion_config
    
    
    def get_base_model_config(self) -> BaseModelConfig:
        config = self.config.base_model
        
        create_directories([config.root_dir])

        base_model_config = BaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return base_model_config