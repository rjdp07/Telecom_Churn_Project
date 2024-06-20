from churnPrediction.config.configuration import ConfigurationManager
from churnPrediction.components.model_evaluation import ModelEvaluation
from churnPrediction import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.log_into_mlfow()



if __name__ == '__main__':
    try:
        logger.info(f">>>>> {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>> {STAGE_NAME} completed <<<<<<\n\nx==============x")

    except Exception as e:
        logger.exception(e)
        raise e
