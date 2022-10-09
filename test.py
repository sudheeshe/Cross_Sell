from Src.stage_3_data_pipeline import DataPipeline
from Src.stage_4_model_training import model_metrics_plots, model_training_XGBClassifier, model_training_BalancedRandomForestClassifier

from xgboost import XGBClassifier

pipeline = DataPipeline()

x_train, x_val, y_train, y_val = pipeline.data_pipeline()

#print(x_train)
#print(x_val)

model_xgb = model_training_XGBClassifier()
model_metrics_plots(model_xgb,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    model_name="XGB_model_report",
                    name = 'XGB_model_report',
                    fig_size=(20, 20))


model_brf = model_training_BalancedRandomForestClassifier()
model_metrics_plots(model_brf,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    model_name="BalancedRandomForestClassifier",
                    name = 'BalancedRandomForestClassifier_model_report',
                    fig_size=(20, 20))

print("Finished...")