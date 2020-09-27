import pandas as pd
from sklearn import preprocessing
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")
import datetime
import traceback



def read_preproc_cut_transcations_data(path='./hack_data/transactions.parquet', 
                                       client_ids=None, plants=None, materials=None,
                                       rows_number=2 * 10**7):
    """
    Read, preprocess transactions.parquet and take first rows_number rows of it.
    """
    trans = pd.read_parquet(path, engine='pyarrow', use_threads=True)
    trans_short = trans.iloc[:rows_number]
    trans_short = trans_short[['plant', 'client_id', 'chq_date', 'sales_count', 'sales_sum', 'material']]
    trans_short = trans_short[trans_short['sales_sum'] >= 0]

    if client_ids is not None:
        trans_short = trans_short[trans_short['client_id'].isin(client_ids)]
    if plants is not None:
        trans_short = trans_short[trans_short['plant'].isin(plants)]
    if materials is not None:
        trans_short = trans_short[trans_short['material'].isin(materials)]
        

    le = preprocessing.LabelEncoder()
    trans_short['client_id'] = le.fit_transform(trans_short['client_id'])
    trans_short['product_id'] = le.fit_transform(trans_short['material'])
    trans_short['shop_id'] = le.fit_transform(trans_short['plant'])
    trans_short = trans_short.drop(columns=['material', 'plant'])
    return trans_short
    

def predict_stock_quantity(data, product_id, shop_id, current_date, prediction_length=14):
    """
    Predicts stock quantity for some random next_days_num from the given current_date,
    given prev_days_num days. Emulates one cycle before a new supply.
    """
    try:
        le = preprocessing.LabelEncoder()
        product_df = data[data['product_id'] == product_id]
        product_df = product_df[product_df['shop_id'] == shop_id]
        product_df = product_df[['chq_date', 'sales_count']]
        product_df = product_df.sort_values(by=['chq_date'])
        product_df = product_df.groupby(['chq_date'], as_index=False).sum()
        product_df['day_number'] = le.fit_transform(product_df['chq_date'])

#         np.random.seed(42)
        prev_days_num = prediction_length #np.random.randint(low=5, high=14)
        next_days_num = prediction_length

        current_day_id = product_df[product_df['chq_date'] <= current_date]['day_number'].values[-1]
    
        week_df = product_df[product_df['day_number'] <= current_day_id + next_days_num]
        week_df = week_df[product_df['day_number'] > current_day_id - prev_days_num]
        week_df['cum_sum'] = np.cumsum(week_df['sales_count'])
        week_df['stock_amount'] = week_df['cum_sum'].apply(lambda x: np.sum(week_df['sales_count']) - x)
        week_df['stock_amount'] += np.random.randint(low=-100, high=100, size=week_df.shape[0])

        train_demand = week_df[week_df['chq_date'] <= current_date]['sales_count'].values[-prev_days_num:]

        model = ExponentialSmoothing(train_demand, trend='mul')
        fit = model.fit()
        predictions = fit.forecast(next_days_num)
        predictions = np.sort(predictions)[::-1]
    except Exception as e:
        print(f"Error on calculation: {e}")
        print(traceback.format_exc())
        predictions = np.zeros(next_days_num)
    return predictions


def predict_demand(data, product_id, shop_id, current_date, prediction_length=14):
    """
    Predict next 14 days of the given product demand in a given market.
    """
    try:
        le = preprocessing.LabelEncoder()
        product_df = data[data['product_id'] == product_id]
        product_df = product_df[product_df['shop_id'] == shop_id]
        product_df = product_df[['chq_date', 'sales_count']]
        product_df = product_df.sort_values(by=['chq_date'])
        product_df = product_df.groupby(['chq_date'], as_index=False).sum()
        product_df['day_number'] = le.fit_transform(product_df['chq_date'])

        train_demand = product_df[product_df['chq_date'] <= current_date]['sales_count'].values

        model = ExponentialSmoothing(train_demand, trend='mul', seasonal='add', seasonal_periods=7)
        fit = model.fit()
        predictions = fit.forecast(prediction_length)
    except Exception as e:
        print(f"Error on calculation: {e}")
        print(traceback.format_exc())
        predictions = np.zeros(prediction_length)
    return predictions




