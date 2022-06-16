import helpers.helper as hh
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#Predict
def prediction(
    clf,
    ss,
    le,
    filter_cols,
    X
):
    if filter_cols:
        X = X.iloc[:,filter_cols]

    x_test = ss.transform(X)

    pred = clf.predict(x_test)
    pred = le.classes_[pred]

    return pred

def main():
    ntp = 5
    tw = 7
    response,log = hh.get_ipfs("bafkreicu7w25tzdmxeghhdnorfjgdv4apk66rnuy53gukjiv75c5smq6ha")
    clf = hh.get_model("bafkreial5n5h72jz5ng2d5e43kzvilccn7g7jt7v4qx7jro4cxa6dpgzpy")
    ss = hh.get_model("bafkreibdshyex5cemi3jzksf3vys5cc4gnmqrggr2yp55gmsorlyiakdv4")
    le = hh.get_model("bafkreibzscgshq42slwii4ney54sqv75mixfzzmq3hyhonqzfzbauwu4hy")

    dataset = hh.read_file(response.content).iloc[-(ntp+tw):]

    df_predict,y = hh.get_tabpandas_multi(dataset.drop(columns=["Date"]),ntp,tw)

    pred = prediction(clf,ss,le,None,df_predict)

    print(pred)

    return pred
if __name__ == "__main__":
    main()
