from flask import Flask, request, render_template
import pandas as pd
import pickle as pkl
import numpy as np



app = Flask(__name__)


@app.route('/', methods=['GET', "POST"])
def home():
    return render_template('home_page.html')

@app.route('/submit', methods=["POST"])
def result():

    #Taking variables from GUI

    gender = str(request.form.get('gender'))
    age = str(request.form.get('age'))
    region_code = float(request.form.get('region_code'))
    previously_insured = int(request.form.get('previously_insured'))
    vehicle_age = str(request.form.get('vehicle_age'))
    vehicle_damage = str(request.form.get('vehicle_damage'))
    annual_premium = float(request.form.get('annual_premium'))
    policy_sales_channel = str(request.form.get('policy_sales_channel'))
    vintage = int(request.form.get('vintage'))


    # Loading pickle files
    cat_encoder = pkl.load(open('D:/Ineuron/Project_workshop/Cross_Selling/Pickle/categorical_encoder.pkl', 'rb'))
    model = pkl.load(open('D:/Ineuron/Project_workshop/Cross_Selling/Models/XGBClassifier.pkl', 'rb'))




    dataframe = pd.DataFrame(
        [[gender, age, region_code, previously_insured, vehicle_age, vehicle_damage,
          annual_premium, policy_sales_channel, vintage]],
        columns= ['Gender', 'Age', 'Region_Code', 'Previously_Insured', 'Vehicle_Age',
                  'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage'])

    df_array = cat_encoder.transform(dataframe)


    result_1 = model.predict(df_array)
    #result_2 = model.predict_proba(df_array)

    output = ""

    if result_1 == 0:
        output= f"Model prediction is {str(result_1)} so the customer will not buy Vehicle insurance"

    elif result_1 == 1:
        output= f"Model prediction is {str(result_1)} so the customer will buy Vehicle insurance"


    return render_template('result.html', result=output)


if __name__ == "__main__":
        app.run(debug=True)