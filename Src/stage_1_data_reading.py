import pandas as pd
from Src.logger import AppLogger


class ReadData:
    """
        Description: "This Module helps in reading csv data
    """

    def __init__(self):
        self.file = open('D:/Ineuron/Project_workshop/Cross_Selling/Logs/Data_reading_log.txt', 'a+')
        self.logger = AppLogger()


    def read_data(self,data):

        """
        Description: This method helps in reading csv
        return:: dataframe
        """

        try:
            self.logger.log(self.file,
                            'Inside read_data method of stage_1 class >>> Started reading the given csv data.')

            dataframe = pd.read_csv(data)
            self.logger.log(self.file, 'Data read successfully. Returning dataframe')
            self.logger.log(self.file, 'Leaving read_data method of stage_1 class')


            return dataframe

        except Exception as e:
            self.logger.log(self.file, str(e))