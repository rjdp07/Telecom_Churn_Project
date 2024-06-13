from churnPrediction.config.configuration import ConfigurationManager
from churnPrediction.components.data_transformation import DataTransformation
from churnPrediction import logger


STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.ColumnConvert()
        data_transformation.XYDataSets()
        data_transformation.ColumnPreProcess()
        data_transformation.ClassBalancing()
        data_transformation.train_test_splitting()



if __name__ == '__main__':
    try:
        logger.info(f">>>>> {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>> {STAGE_NAME} completed <<<<<<\n\nx==============x")

    except Exception as e:
        logger.exception(e)
        raise e
