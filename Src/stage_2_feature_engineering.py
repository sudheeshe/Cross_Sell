import pandas as pd
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, KBinsDiscretizer
from Src.logger import AppLogger


class FeatureEngineering:

    def __init__(self):
        self.logger = AppLogger()
        self.file = self.file = open('D:/Ineuron/Project_workshop/Cross_Selling/Logs/FeatureEngg_logs.txt', 'a+')

    def drop_columns(self, data, columns, axis='columns'):

        """
        Description: This method helps in dropping the specified columns
        return: dataframe
        """
        try:
            self.logger.log(self.file,
                            f'Inside drop_columns method of stage_2 class >>> Started dropping the {columns} columns from dataset')

            temp_df = data.drop(columns=columns, axis=axis)

            self.logger.log(self.file, f'Dropping {columns} columns were successful, returning dataframe.')
            self.logger.log(self.file, 'Leaving drop_columns method of stage_2 class')


            return temp_df

        except Exception as e:
            self.logger.log(self.file, str(e))


    def categorize_age(self, data, column):

        """
        Description: This method helps to make age categories.

        params: data - dataset
                columns - columns to perform merging
        return: dataframe
        """

        try:
            self.logger.log(self.file,
                            f'Inside categorize_age method of stage_2 class >>> Making age category started')

            data[column] = data[column].astype(str)

            data[column] = np.where(data[column].between('15', '20'), 'Adolescence', data[column])
            data[column] = np.where(data[column].between('21', '30'), 'Early_adulthood', data[column])
            data[column] = np.where(data[column].between('31', '39'), 'Mid_life', data[column])
            data[column] = np.where(data[column].between('40', '65'), 'Mature_adulthood', data[column])
            data[column] = np.where(data[column].between('66', '90'), 'Late_adulthood', data[column])

            self.logger.log(self.file,
                            f'Making age category were successful, returning dataframe.')
            self.logger.log(self.file, 'Leaving categorize_age method of stage_2 class')

            return data

        except Exception as e:
            self.logger.log(self.file, str(e))



    def top_categories(self, data, column, top_ = 10, name_to_replace= 'other'):

        """
        Description: This method helps in merging less frequent categories to a single category.
        params: data - dataset
                columns - columns to perform merging
                top_ =  how may top categories to select, default is top 10 categories
                name_to_replace = the new class name default is other
        return: dataframe
        """

        try:
            self.logger.log(self.file,
                            f'Inside top_categories method of stage_2 class >>> Performing category merging of less frequent categories on columns {column}')

            top_catgry = list(data[column].value_counts().head(top_).index)

            self.logger.log(self.file,
                            f'Found {top_catgry} categories which are occurring very frequently')

            data[column] = np.where(data[column].isin(top_catgry), data[column], name_to_replace)

            self.logger.log(self.file,
                            f'Merging of less frequent classes on {column} columns were successful, returning dataframe.')
            self.logger.log(self.file, 'Leaving top_categories method of stage_2 class')

            return data

        except Exception as e:
            self.logger.log(self.file, str(e))




    def categorical_encoder(self, data):

        """
         Description: This method helps in encoding categorical variable.
         return: dataframe
        """

        try:
            self.logger.log(self.file,
                            'Inside categorical_encoder method of stage_2 class >>> Making Column transformer for ordinal encoding on "Age, Vehicle_Age" columns .')

            self.logger.log(self.file,
                            'Inside categorical_encoder method of stage_2 class >>> Making Column transformer for One Hot encoding on "Gender, Previously_Insured, Vehicle_Damage, Policy_Sales_Channel" columns.')

            self.logger.log(self.file,
                        'Inside categorical_encoder method of stage_2 class >>> Making Column transformer for Binning on "Annual_Premium, Vintage" columns using sklearn.KBinsDiscretizer.')

            col_transformer = ColumnTransformer([('ordinal_encoder', OrdinalEncoder(
                categories=[['Adolescence', 'Early_adulthood', 'Mid_life', 'Mature_adulthood', 'Late_adulthood'],
                            ['< 1 Year', '1-2 Year', '> 2 Years']], dtype=np.int64),
                                                  ['Age', 'Vehicle_Age']),

                                                 ('one_hot_encoder', OneHotEncoder(categories='auto', drop='first'),
                                                  ['Gender', 'Previously_Insured', 'Vehicle_Damage',
                                                   'Policy_Sales_Channel']),

                                                 ('binning_region_code',
                                                  KBinsDiscretizer(n_bins=7, strategy='quantile', encode='onehot-dense'),
                                                  ['Region_Code']),

                                                 ('binning_annual_premium',
                                                  KBinsDiscretizer(n_bins=6, encode='onehot-dense', strategy='kmeans'),
                                                  ['Annual_Premium']),

                                                 ('binning_vintage',
                                                  KBinsDiscretizer(n_bins=10, strategy='quantile', encode='onehot-dense'),
                                                  ['Vintage'])], remainder='passthrough'

                                                )

            self.logger.log(self.file,
                            ' Successfully made column transformer with Ordinal, OneHot encoder and KBinsDiscretizer, ready to apply on data.')

            encoder = col_transformer.fit(data)

            temp_df = encoder.fit_transform(data)

            filename = 'D:/Ineuron/Project_workshop/Cross_Selling/Pickle/categorical_encoder.pkl'
            pkl.dump(encoder, open(filename, 'wb'))

            self.logger.log(self.file,
                            'Saved the column transformer with Ordinal and OneHot encoder as categorical_encoder.pkl in Pickle folder, returning the transformer')
            self.logger.log(self.file, 'Leaving categorical_encoder method of stage_2 class')

            temp_df = pd.DataFrame(temp_df)

            return temp_df

        except Exception as e:
            self.logger.log(self.file, str(e))



    def label_encoder(self, data):

        """
             Description: This method helps in Label encoding of target column.
             return: dataframe
        """

        try:
            self.logger.log(self.file,
                            'Inside Label_encoder method of stage_2 class >>> Starting the label encoding')

            encoder = LabelEncoder()
            encoder.fit(data)

            filename = 'D:/Ineuron/Project_workshop/Cross_Selling/Pickle/label_encoder.pkl'
            pkl.dump(encoder, open(filename, 'wb'))

            self.logger.log(self.file,
                            'Saved the label_encoder as label_encoder.pkl in Pickle folder,')

            array = encoder.fit_transform(data)

            self.logger.log(self.file,
                            f' Label encoding on target column was successful, returning array.')

            self.logger.log(self.file, 'Leaving Label_encoder method of stage_2 class')

            temp_df = pd.DataFrame(array)

            return temp_df

        except Exception as e:
            self.logger.log(self.file, str(e))


    def split_save_processed_data(self, X, Y, val_split_size=0.2):

        try:
            self.logger.log(self.file,
                            'Inside split_save_processed_data method of stage_2 class >>> Starting the splitting data into train and validation set')
            x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=val_split_size, random_state=369)

            x_train.to_csv('D:/Ineuron/Project_workshop/Cross_Selling/Data/processed_data/x_train.csv', index=False)
            y_train.to_csv('D:/Ineuron/Project_workshop/Cross_Selling/Data/processed_data/y_train.csv', index=False)

            x_val.to_csv('D:/Ineuron/Project_workshop/Cross_Selling/Data/processed_data/x_val.csv', index=False)
            y_val.to_csv('D:/Ineuron/Project_workshop/Cross_Selling/Data/processed_data/y_val.csv', index=False)

            self.logger.log(self.file,
                            f' Splitting data into train and validation set was successful, returning dataframes x_train, x_val, y_train, y_val.')

            self.logger.log(self.file, 'Leaving split_save_processed_data method of stage_2 class')

            return  x_train, x_val, y_train, y_val


        except Exception as e:
            self.logger.log(self.file, str(e))