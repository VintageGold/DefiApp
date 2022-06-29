from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

def plot_confusion_matrix(
    targs,
    preds,
    title
):
    cm = confusion_matrix(targs, preds, normalize=None)
    df_cm = pd.DataFrame(cm,
                         columns=['DAI', 'USDC', 'USDT'],
                         index=['DAI', 'USDC', 'USDT',])

    ax = sns.heatmap(df_cm,
            annot=True,
            annot_kws={"size": 16},
            cmap="Blues",
            ).set_title(title)

    plt.show()


#Predict
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
def prediction(
    clf,
    ss,
    le,
    X,
    y,
    filter_cols
):
    if filter_cols:
        X = X.iloc[:,filter_cols]

    x_test = ss.transform(X)

    pred = clf.predict(x_test)
    pred = le.classes_[pred]

    print(f"Accuracy score: {accuracy_score(y, pred)}")

    return (pred,accuracy_score(y, pred),f1_score(y,pred,average="micro"),
        precision_score(y,pred,average="micro"),
        recall_score(y,pred,average="micro"))

#dollar impact
def lowest_cost(row):
    return row[row.Lowest_br_cost] * row['Borrow Amount']/365
#dollar impact
def strategy_cost(row):
    return row[row.Predict] * row['Borrow Amount']/365

def compare_strategy(ntp, tw, initial_borrow, df,y,ss,le,clf,filter_cols):

    chunks = []
    for i, v in enumerate(range(0, len(df), tw)):
        if i == 0:
            chunks.append(df.iloc[:ntp+tw].copy().reset_index(drop=True))
        else:
            chunks.append(df.iloc[v:v+ntp+tw].copy().reset_index(drop=True))

    for i, chu in enumerate(chunks):
        if len(chu) == ntp+tw:

            if filter_cols:
                x = chu.iloc[:,filter_cols]
                x = x.iloc[:ntp,:]

            else:
                x = chu.iloc[:ntp,:]

            x = ss.transform(x)
            pred = clf.predict(x)
            pred = le.classes_[pred]

            chu.loc[ntp:, 'Predict'] = pred[0]

            if i == 0:
                final = chu.dropna()
            else:
                final = final.append(chu.dropna()).reset_index(drop=True)


    final = (final[['DAI_borrowRate_t-0','USDC_borrowRate_t-0','USDT_borrowRate_t-0','Predict']]
             .rename(columns={"DAI_borrowRate_t-0":"DAI_borrowRate",
                      'USDC_borrowRate_t-0':'USDC_borrowRate',
                      'USDT_borrowRate_t-0':'USDT_borrowRate'})
)
    final['Lowest_br_cost'] = y
    final['Borrow Amount'] = initial_borrow
    final['DAI_br_cost'] = final['DAI_borrowRate'] * final['Borrow Amount']/365
    final['USDC_br_cost'] = final['USDC_borrowRate'] * final['Borrow Amount']/365
    final['USDT_br_cost'] = final['USDT_borrowRate'] * final['Borrow Amount']/365


    final['Strategy_br_cost'] = final.apply(lambda row: strategy_cost(row), axis=1)
    final['Lowest_br_cost'] = final.apply(lambda row: lowest_cost(row), axis=1)

    return (final[['Borrow Amount','DAI_br_cost','USDC_br_cost','USDT_br_cost','Strategy_br_cost','Lowest_br_cost']],
            len(chunks)
            )
