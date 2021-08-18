#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import catboost as cb
from sklearn.preprocessing import LabelEncoder
import pickle

#Initialize the flask App
app = Flask(__name__)
cb_model = cb.CatBoostRegressor(loss_function='RMSE')
cb_model.load_model('cb_model_5')

pkl_file = open('source_encoder5.pkl', 'rb')
le_source = pickle.load(pkl_file) 
pkl_file.close()

pkl_file = open('country_encoder5.pkl', 'rb')
le_country = pickle.load(pkl_file) 
pkl_file.close()


pkl_file = open('partner_encoder5.pkl', 'rb')
le_partner = pickle.load(pkl_file) 
pkl_file.close()

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

def create_date_featues(df):
    df['Year'] = pd.to_datetime(df['import_date']).dt.year
    df['Month'] = pd.to_datetime(df['import_date']).dt.month
    df['Day'] = pd.to_datetime(df['import_date']).dt.day
    df['Dayofweek'] = pd.to_datetime(df['import_date']).dt.dayofweek
    df['DayOfyear'] = pd.to_datetime(df['import_date']).dt.dayofyear
    df['Week'] = pd.to_datetime(df['import_date']).dt.week 
    df['Quarter'] = pd.to_datetime(df['import_date']).dt.quarter  
    df['Is_month_start'] = pd.to_datetime(df['import_date']).dt.is_month_start 
    df['Is_month_end'] = pd.to_datetime(df['import_date']).dt.is_month_end 
    df['Is_quarter_start'] = pd.to_datetime(df['import_date']).dt.is_quarter_start
    df['Is_quarter_end'] = pd.to_datetime(df['import_date']).dt.is_quarter_end 
    df['Is_year_start'] = pd.to_datetime(df['import_date']).dt.is_year_start 
    df['Is_year_end'] = pd.to_datetime(df['import_date']).dt.is_year_end
    df['Semester'] = np.where(df['Quarter'].isin([1,2]),1,2)
    df['Is_weekend'] = np.where(df['Dayofweek'].isin([5,6]),1,0)
    df['Is_weekday'] = np.where(df['Dayofweek'].isin([0,1,2,3,4]),1,0)
    df['Days_in_month'] = pd.to_datetime(df['import_date']).dt.days_in_month

    return df

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]

    
    source = int(features[0])
    country = features[1]
    partner = features[2]
    packet_date = features[3]

    import_dates = pd.date_range(packet_date, periods=5, freq='D')
    df = pd.DataFrame()
    df['source'] = [source]*5
    df['country'] = [country]*5
    df['partner'] = [partner]*5
    df['packet_date'] = [packet_date]*5
    df['import_date'] = import_dates

    df['packet_date'] = pd.to_datetime(df['packet_date'])
    df['import_date'] = pd.to_datetime(df['import_date'])
    df['day_num'] = df['import_date'] - df['packet_date'] 
    df['day_num'] = pd.to_numeric(df['day_num'].dt.days) + 1

    cols = ['packet_date']
    for col in cols:
        del df[col]

    df = df[['import_date', 'source', 'country', 'partner', 'day_num']]

    df['import_date'] = pd.to_datetime(df['import_date'])

    df = create_date_featues(df)

    one_hot_cols = ['Is_month_start',
     'Is_month_end',
     'Is_quarter_start',
     'Is_quarter_end',
     'Is_year_start',
     'Is_year_end']

    t_f = {True:1, False:0}
    def map_column(df, cols, t_f):
        for col in cols:
            df[col] = df[col].map(t_f)
        return df
    df = map_column(df, one_hot_cols, t_f)

    
     

    df['source'] = le_source.transform(df['source'])
    df['country'] = le_country.transform(df['country'])
    df['partner'] = le_partner.transform(df['partner'])


 
    del df['import_date']


    pred = cb_model.predict(df)
    pred = np.clip(pred, a_min=0, a_max=200)

    dff = pd.DataFrame()
    dff['DayNumber'] = [i for i in range(1,6)]
    dff['EPM'] = np.expm1(pred)
    dff.set_index('DayNumber', inplace=True)
    s = dff['EPM'].sum()









    return render_template('index.html', data = dff.to_html())

if __name__ == "__main__":
    app.run(debug=True)