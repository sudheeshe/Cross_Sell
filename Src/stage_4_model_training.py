import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from Src.logger import AppLogger
from Src.stage_3_data_pipeline import DataPipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import auc, roc_curve, confusion_matrix, classification_report, roc_auc_score, precision_recall_curve
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay


params_brf = [{'n_estimators': [100,200,300,500,1000],
         'class_weight': ['balanced', 'balanced_subsample'],
         'criterion': ['gini', 'entropy'],
         'max_depth': [20,50,70,100, 150, 200, 300, 500]}]

params_xgb = {'n_estimators': np.random.randint(100, 300, 15),
               'booster': ['gbtree', 'dart'],
               'scale_pos_weight': np.random.randint(1, 1000, 5)}



cv = StratifiedKFold(n_splits=5)

def model_tuning(model, x_train,y_train, params, cv):


    train_model = RandomizedSearchCV(model, param_distributions=params, n_iter=30, cv=cv, verbose=True)
    best_clf = train_model.fit(x_train, y_train)
    best_train_score = best_clf.best_score_
    best_model_params = best_clf.best_params_

    return best_train_score, best_model_params


def model_training_BalancedRandomForestClassifier():
    try:
        logger = AppLogger()
        file = open('D:/Ineuron/Project_workshop/LeadScore/Logs/ModelTraining_logs.txt', 'a+')

        ##importing data_pipeline method
        pipeline = DataPipeline()
        x_train, x_val, y_train, y_val = pipeline.data_pipeline()

        logger.log(file, "x_train & y_train dataframes read successfully")

        model = BalancedRandomForestClassifier(n_estimators= 500,
                                               max_depth=500,
                                               criterion='gini',
                                               class_weight='balanced_subsample')
        model.fit(x_train, y_train)

        logger.log(file, "BalancedRandomForestClassifier trained successfully")

        filename = 'D:/Ineuron/Project_workshop/Cross_Selling/Models/BalancedRandomForestClassifier.pkl'
        pkl.dump(model, open(filename, 'wb'))

        logger.log(file, "BalancedRandomForestClassifier model saved successfully")
        return model

    except Exception as e:
        print(e)




def model_training_XGBClassifier():

    try:
        logger = AppLogger()
        file = open('D:/Ineuron/Project_workshop/LeadScore/Logs/ModelTraining_logs.txt', 'a+')

        ##importing data_pipeline method
        pipeline = DataPipeline()
        x_train, x_val, y_train, y_val = pipeline.data_pipeline()

        logger.log(file, "x_train & y_train dataframes read successfully")

        model = XGBClassifier(scale_pos_weight=261, n_estimators=286, booster='gbtree')
        model.fit(x_train, y_train)

        logger.log(file, "XGBClassifier trained successfully")

        filename = 'D:/Ineuron/Project_workshop/Cross_Selling/Models/XGBClassifier.pkl'
        pkl.dump(model, open(filename, 'wb'))

        logger.log(file, "XGBClassifier model saved successfully")
        return model

    except Exception as e:
        print(e)


def model_metrics_plots(model, x_train, y_train, x_val, y_val, model_name="XGBoost", fig_size=(15, 15), name = 'XGB_model_report'):
    model.fit(x_train, y_train)
    pred_y_train = model.predict(x_train)
    pred_y_val = model.predict(x_val)

    fig, axes = plt.subplots(4, 2, figsize=fig_size)
    fig.suptitle(f"{model_name} Model Metrics")

    ConfusionMatrixDisplay.from_estimator(model, x_train, y_train, ax=axes[0, 0])
    axes[0, 0].set_title('Confusion matrix for Training Data')
    axes[0, 0].grid(False)

    ConfusionMatrixDisplay.from_estimator(model, x_val, y_val, ax=axes[0, 1])
    axes[0, 1].set_title('Confusion matrix for Validation Data')
    axes[0, 1].grid(False)

    ConfusionMatrixDisplay.from_estimator(model, x_train, y_train, ax=axes[1, 0], normalize='true')
    axes[1, 0].set_title('Normalized Confusion matrix for Training Data')
    axes[1, 0].grid(False)

    ConfusionMatrixDisplay.from_estimator(model, x_val, y_val, ax=axes[1, 1], normalize='true')
    axes[1, 1].set_title('Normalized Confusion matrix for Validation Data')
    axes[1, 1].grid(False)

    RocCurveDisplay.from_estimator(model, x_train, y_train, ax=axes[2, 0])
    axes[2, 0].set_title('ROC for Training Data')

    RocCurveDisplay.from_estimator(model, x_val, y_val, ax=axes[2, 1])
    axes[2, 1].set_title('ROC for Validation Data')

    PrecisionRecallDisplay.from_estimator(model, x_train, y_train, ax=axes[3, 0])
    axes[3, 0].set_title('Precsion-Recall curve for Training Data')

    PrecisionRecallDisplay.from_estimator(model, x_val, y_val, ax=axes[3, 1])
    axes[3, 1].set_title('Precsion-Recall curve for Validation Data')

    print("Classification Report of Training data\n")
    print(classification_report(y_train, pred_y_train))
    print("Classification Report of Validation data\n")
    print(classification_report(y_val, pred_y_val))

    fig.savefig(f'D:/Ineuron/Project_workshop/Cross_Selling/Reports/{name}.png')




if __name__ == "__main__":
    pipeline = DataPipeline()

    x_train, x_val, y_train, y_val = pipeline.data_pipeline()

    # print(x_train)
    # print(x_val)

    model_xgb = model_training_XGBClassifier()
    model_metrics_plots(model_xgb,
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        model_name="XGB_model_report",
                        name='XGB_model_report',
                        fig_size=(20, 20))

    model_brf = model_training_BalancedRandomForestClassifier()
    model_metrics_plots(model_brf,
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        model_name="BalancedRandomForestClassifier",
                        name='BalancedRandomForestClassifier_model_report',
                        fig_size=(20, 20))

    print("Finished...")
