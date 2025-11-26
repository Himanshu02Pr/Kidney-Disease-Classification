from Classifier.config.configuration import ConfigurationManager
from Classifier.components.model_evaluation_mlflow import Evaluation
from Classifier import logger



STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow()




if __name__ == '__main__':
    try:
        logger.info(f"\n\n<-------- {STAGE_NAME} started -------->\n\n")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f"\n\n<-------- {STAGE_NAME} completed -------->\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
            