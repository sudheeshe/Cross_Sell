from Src.stage_1_data_reading import ReadData
from Src.stage_2_feature_engineering import FeatureEngineering
from Src.logger import AppLogger


class DataPipeline:

    def __init__(self):
        self.logger = AppLogger()
        self.file = self.file = open('D:/Ineuron/Project_workshop/LeadScore/Logs/DataPipeline_logs.txt','a+')

    def data_pipeline(self):
        try:
            """
              Description: This method helps in data manipulation like data cleaning and feature engg
              return: dataframe
            """
            self.logger.log(self.file,
                            f'Inside data_pipeline method of stage_3 class >>> Started data preprocessing ')

            #################################### Reading Data #########################################

            data = 'D:/Ineuron/Project_workshop/Cross_Selling/Data/raw_data/train.csv'
            read = ReadData()
            df = read.read_data(data)

            #################################### Feature Engg #########################################

            fe = FeatureEngineering()
            df_feature_engg = fe.drop_columns(df, columns=['id', 'Driving_License'])

            # Dropping columns
            df_age_cat = fe.categorize_age(df_feature_engg, 'Age')

            #Grouping rare categories
            df_top_10 = fe.top_categories(df_age_cat, 'Policy_Sales_Channel', top_=10, name_to_replace='other')

            # Splitting data to independent and dependent variable
            X = df_top_10.drop(columns=['Response'], axis='columns')
            Y = df_top_10['Response']

            # Applying column transformer
            df_encoded = fe.categorical_encoder(X)
            label_encoded = fe.label_encoder(Y)

            # Splitting the data to training and validation dataset
            x_train, x_val, y_train, y_val = fe.split_save_processed_data(df_encoded, label_encoded, val_split_size=0.2)

            self.logger.log(self.file, "Successfully completed Feature engineering on the dataset.")
            self.logger.log(self.file,
                            "Saving the dataset @ Data/processed_data/x_train.csv, Data/processed_data/y_train.csv. Data/processed_data/x_val.csv, Data/processed_data/y_val.csv")

            self.logger.log(self.file, "Returning dataframes x_train, x_val, y_train, y_val.")
            self.logger.log(self.file, 'Leaving data_pipeline method of stage_3 class')

            return x_train, x_val, y_train, y_val

        except Exception as e:
            self.logger.log(self.file, str(e))