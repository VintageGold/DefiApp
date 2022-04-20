from helpers.helper import read_file,remove_missing_timestamps,prepare_for_modeling,get_tabpandas_multi,agg_to_day
import pickle
from io import BytesIO
from storage.ipfs import IPFS
import streamlit as st
import pandas as pd

import time

class Retry(object):
    default_exceptions = (Exception,)
    def __init__(self, tries, exceptions=None, delay=0):
        """
        Decorator for retrying a function if exception occurs

        tries -- num tries
        exceptions -- exceptions to catch
        delay -- wait between retries
        """
        self.tries = tries
        if exceptions is None:
            exceptions = Retry.default_exceptions
        self.exceptions =  exceptions
        self.delay = delay

    def __call__(self, f):
        def fn(*args, **kwargs):
            exception = None
            for _ in range(self.tries):
                try:
                    return f(*args, **kwargs)
                except self.exceptions as e:
                    print("Retry, exception: "+str(e))
                    time.sleep(self.delay)
                    exception = e
            #if no success after tries, raise last exception
            raise exception
        return fn

def get_success(msg):
    return st.success(msg)

def get_info(msg):
    return st.info(msg)

def get_ipfs(cid,cred=None):
    ipfs = IPFS()

    response, log = ipfs.get_file(cid,local_node=False)

    # st.write(type(response.status_code),log)

    if response.status_code != 200:
        get_ipfs(cid)

    if response.status_code == 200:
        return response, log

def _prep_testdata(df):

    df = remove_missing_timestamps(df)

    df = prepare_for_modeling(df)

    df.sort_values('Timestamp', inplace=True)

    df["Date"] = pd.to_datetime(df["Timestamp"], unit='s', origin='unix')

    df = agg_to_day(df)

    return df.iloc[:, 1:]

@Retry(5)
def get_data(cid):

    if cid:
        # get_info("Retrieving Dataset")
        response,log = get_ipfs(cid)


        df = read_file(response.content)

        # get_success("Loaded Dataset")

        df_prep = _prep_testdata(df)

        return df_prep

@Retry(5)
def get_model(cid,deep=False):

    if cid:

        # get_info("Retrieving Model")

        response, log = get_ipfs(cid)

        string_of_bytes_obj = str(pickle.dumps(response.content), encoding="latin1")

        unpickled_dict = pickle.loads(bytes(string_of_bytes_obj, "latin1"))

        if deep:
            pass
            # inference = torch.load(BytesIO(unpickled_dict),map_location=torch.device('cpu'))
        else:
            inference = pickle.load(BytesIO(unpickled_dict))

        # get_success("Loaded Model")



        return inference
