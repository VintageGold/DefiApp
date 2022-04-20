from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from fastai.tabular.all import *
import numpy as np



def plot_confusion_matrix(
    targs,
    preds
        ):
        cm = confusion_matrix(targs, preds, normalize='true')
        df_cm = pd.DataFrame(cm,
                             columns=['DAI', 'USDC', 'USDT'],
                             index=['DAI', 'USDC', 'USDT'])

        fig,axes = plt.subplots(1, 1,figsize=(4,4))

        sns.heatmap(df_cm,
                annot=True,
                annot_kws={"size": 16},
                cmap="Blues",
                )
        st.pyplot(fig)

def strategy_cost(row):
    return row[row.Classification]

def br_plot(init_amount,model,ss,le,DF,ntp,tw,only_model=False):

    DF["Timestamp"] = pd.to_datetime(DF["Timestamp"], unit='s', origin='unix')

    chunks = []
    for i, v in enumerate(range(0, len(DF), 7)):
        if i == 0:
            chunks.append(DF.loc[:ntp+tw-1].copy().reset_index(drop=True))
        else:
            chunks.append(DF.loc[v:v+ntp+tw-1].copy().reset_index(drop=True))

    for i, chu in enumerate(chunks):
        x = np.array(chu.iloc[:ntp, 1:].values.reshape(1, -1))
        x = ss.transform(x)
        pred = model.predict(x)
        pred = le.classes_[pred]

        chu.loc[ntp:, 'Classification'] = pred[0]
        if i == 0:
            final = chu.dropna()
        else:
            final = final.append(chu.dropna()).reset_index(drop=True)


    if only_model:

            final['Borrow Amount'] = init_amount
            final['M2_Borrowing Rate'] = final.apply(lambda row: strategy_cost(row), axis=1)
            final['M2_br_cost'] = (final['M2_Borrowing Rate']/365) * final['Borrow Amount']
            final['Date'] = DF["Timestamp"]

            return final

    final['Borrow Amount'] = init_amount
    final['DAI_br_cost'] = (final['DAI_Borrowing Rate']/365) * final['Borrow Amount']
    final['USDC_br_cost'] = (final['USDC_Borrowing Rate']/365) * final['Borrow Amount']
    final['USDT_br_cost'] = (final['USDT_Borrowing Rate']/365) * final['Borrow Amount']
    final['M1_Borrowing Rate'] = final.apply(lambda row: strategy_cost(row), axis=1)
    final['M1_br_cost'] = (final['M1_Borrowing Rate']/365) * final['Borrow Amount']
    final['Date'] = DF["Timestamp"]

    return final[["Classification","M1_br_cost","DAI_br_cost","USDC_br_cost","USDT_br_cost","Date"]]


class SklearnEnsemble:

    def predict(self,init_amount,model,scaler,le,DF,NTP,TW,only_model):#model,scaler,label_encoder
        # x_test_t = scaler.transform(x_test)
        # preds = model.predict(x_test_t)
        # preds = label_encoder.inverse_transform(preds)

        # plot_confusion_matrix(y_test,preds)
        return br_plot(init_amount,model,scaler,le,DF,NTP,TW,only_model)
