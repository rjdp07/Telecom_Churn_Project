from churnPrediction.config.configuration import ConfigurationManager
from churnPrediction.components.model_trainer import ModelTrainer
from churnPrediction import logger


STAGE_NAME = "Model Trainer Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config = model_trainer_config)
        model_trainer_config.train()



if __name__ == '__main__':
    try:
        logger.info(f">>>>> {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> {STAGE_NAME} completed <<<<<<\n\nx==============x")

    except Exception as e:
        logger.exception(e)
        raise e
