from io import StringIO
import pandas as pd
import random
import requests
import pickle
from io import BytesIO
import streamlit as st
#Load file from IPFS

@st.cache
def get_ipfs(cid,local_node=False):
    params = (('arg', cid),)

    #Sourced - All had green checkmarks as of 03/08/2022
    #https://ipfs.github.io/public-gateway-checker/
    gateways = ["https://infura-ipfs.io","https://cf-ipfs.com","https://dweb.link","https://astyanax.io"]

    random.shuffle(gateways)

    log = []

    for gateway in gateways:

        response = requests.get(f"{gateway}/ipfs/{cid}")

        if response.status_code == 200:

            print("Retrieved file hash",cid,f"from {gateway}","Response",response.status_code)

            return response, log

        log.append(gateway)


def get_model(cid,deep=False):

    response, log = get_ipfs(cid)

    string_of_bytes_obj = str(pickle.dumps(response.content), encoding="latin1")

    unpickled_dict = pickle.loads(bytes(string_of_bytes_obj, "latin1"))

    inference = pickle.load(BytesIO(unpickled_dict))

    return inference



def read_file(b):

    s=str(b,'utf-8')

    data = StringIO(s)

    df=pd.read_csv(data)

    return df

#Process into timeseries
def get_tabpandas_multi(
    df:pd.DataFrame, # Dataframe of the raw data
    n_timepoint:int, # Number of previous timepoints to be used as features
    target_window:int, # Number of timepoints in the future to predict
):

    df = df.reset_index(drop=True)
    feature_cols = df.columns

    target_columns = ['DAI_borrowRate', 'USDC_borrowRate', 'USDT_borrowRate']
    target = 'Target'

    cols_names = []
    for j in range(n_timepoint):
        for col in feature_cols:
            cols_names.append(f'{col}_t-{n_timepoint -j-1}')
    cols_names += [target]

    pairs = []
    for i, row in df.iterrows():
        if i < (len(df)-target_window-n_timepoint+1):#+1 bc includes last full prediction set
            features = df.loc[i:i+n_timepoint-1, feature_cols].values #-1 bc loc is inclusive
            features = [item for sublist in features for item in sublist]

            val =  df.loc[i+n_timepoint: i+n_timepoint-1+target_window, target_columns].mean().idxmin()

            features += [val]

            pairs.append(features)

    df = pd.DataFrame(pairs, columns=cols_names).dropna().reset_index(drop=True)


    return df.iloc[:, :-1],df.iloc[:, -1]
