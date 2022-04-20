import streamlit as st
from io import StringIO
import pandas as pd

def read_file(b):

    s=str(b,'utf-8')

    data = StringIO(s)

    df=pd.read_csv(data)

    return df

# resample to daily data - here using mean value
#(other possible way would be to
#use High, Low, Open, Close or using what VG mentioned)
# for now lets keep it simple
def agg_to_day(df):
    return df.resample('D', on='Date').mean().reset_index()

def remove_missing_timestamps(df):
    df["Date"] = pd.to_datetime(df["Timestamp"], unit='s', origin='unix')
    df = df.drop_duplicates(['Timestamp', 'Token'])
    counts = pd.DataFrame(df['Timestamp'].value_counts()).reset_index()
    counts.columns = ['Timestamp', 'Counts']
    df = df.merge(counts, on='Timestamp')

    df = df[df['Counts'] == 4].reset_index(drop=True).drop('Counts', axis=1)

    return df


def prepare_for_modeling(df):
     #make df such that there is one row for each time stamp
    tokens = df["Token"].unique()
    df1 = pd.DataFrame()
    for tok in tokens:
        df_tok = df[df['Token']==tok]
        df_tok = df_tok.drop(['Token', 'Date'], axis=1)

        col_names = []
        for col in df_tok.columns:
            if col == 'Timestamp':
                col_names.append(f'{col}')
            else:
                col_names.append(f'{tok}_{col}')

        df_tok.columns = col_names
        #df_tok = df_tok.set_index('Timestamp', drop=True)

        if df1.empty:
            df1 = df_tok
        else:
            df1 = pd.merge(df1, df_tok, on='Timestamp')

    df1.sort_values('Timestamp', inplace=True)
    df1["Date"] = pd.to_datetime(df1["Timestamp"], unit='s', origin='unix')

    return df1

def get_tabpandas_multi(
    df:pd.DataFrame, # Dataframe of the raw data
    n_timepoint:int, # Number of previous timepoints to be used as features
    target_window:int, # Number of timepoints in the future to predict
    inference:bool=False, # Flag True for inference
    test_size:float=0.20
    ):

    df = df.reset_index(drop=True)
    feature_cols = ['DAI_Borrowing Rate', 'DAI_Deposit Rate', 'DAI_Borrow Volume', 'DAI_Supply Volume',
                    'USDC_Borrowing Rate', 'USDC_Deposit Rate', 'USDC_Borrow Volume', 'USDC_Supply Volume',
                    'USDT_Borrowing Rate', 'USDT_Deposit Rate', 'USDT_Borrow Volume', 'USDT_Supply Volume',
                    'ETH_Borrowing Rate', 'ETH_Deposit Rate', 'ETH_Borrow Volume', 'ETH_Supply Volume'
                    ]

    target_columns = ['DAI_Borrowing Rate', 'USDC_Borrowing Rate', 'USDT_Borrowing Rate']
    target = 'Target'

    cols_names = []
    for j in range(n_timepoint):
        for col in feature_cols:
            cols_names.append(f'{col}_t-{n_timepoint -j-1}')
    cols_names += [target]

    pairs = []
    for i, row in df.iterrows():
        if i < (len(df)-target_window-n_timepoint-1):
            features = df.loc[i:i+n_timepoint-1, feature_cols].values
            features = [item for sublist in features for item in sublist]

            val =  df.loc[i+n_timepoint: i+n_timepoint-1+target_window, target_columns].mean().idxmin()

            features += [val]
            pairs.append(features)

    df = pd.DataFrame(pairs, columns=cols_names).dropna().reset_index(drop=True)

    return df
