import streamlit as st
from ui import create_header,create_footer,create_progressbar
import pandas as pd
from plotting import cumulative_interest,borrowing_rates
from prediction import prediction,compare_strategy
import plotly.express as px
import helpers.helper as hh
import requests


# set variables in session state
st.session_state.ntp = 5

def get_success(msg):
    return st.success(msg)

def get_info(msg):
    return st.info(msg)

def format_string(initial_amount):

    return f"${initial_amount:,}"

def predict(initial_amount,model,scaler,le,DF,NTP,TW,only_model=True):
    sklearn_model = SklearnEnsemble()
    preds = sklearn_model.predict(initial_amount,model,scaler,le,DF,NTP,TW,only_model)
    return preds

def br_cost_ds(date,model,model_cid,accuracy,borrowing_cost,pred):
    temp_df = pd.DataFrame()
    temp_df["Date"] = date[:len(pred)]
    temp_df["Strategy"] = model
    temp_df["Model_cid"] = model_cid
    temp_df["Accuracy"] = accuracy * 100
    temp_df["Borrowing Cost"] = borrowing_cost
    temp_df["Predictions"] = pred

    return temp_df

def combine_datasets():

    #bafkreidnuulqq7cysvb35agjrn2st3nmhprjvp4eooew6z5xqyyigefav4
    response = requests.get("https://thisisstupid.eth.link")

    df_historical = hh.read_file(response.content)

    df_realtime = pd.read_csv("../data/compoundV2.csv")

    dataset = pd.concat([df_historical,df_realtime],axis=0).sort_values("Date",ascending=True)

    return dataset

def main():
    create_header()

    chart1,chart2= st.columns((2,2))

    with chart1.expander("Interest Cost",expanded=True):

        initial_amounts = sorted(range(int(10e6),int(10e7)+int(10e6),int(10e6)),reverse=True)
        initial_amount = st.selectbox("Enter Loan Amount",
                                    options=initial_amounts,
                                    format_func=format_string)

        tw = st.selectbox("Time Window (Days)",options=[7,14,21])

        st.write("Model Load Progress")
        p_bar,progress = create_progressbar()

    #Get Data

    dataset = combine_datasets()

    df_predict,y = hh.get_tabpandas_multi(dataset.drop(columns=["Date"]),st.session_state.ntp,tw)

    load_ipfs_cids = pd.read_csv(f"models/V1/{st.session_state.ntp}_{tw}_models.csv").iloc[:4]

    iter_ipfs_cids = (load_ipfs_cids[["ipfs_pin_hash","keyvalues_scalerLabelEncoder",
                        "keyvalues_datasetTraining"]]
                        .to_numpy().tolist()
    )

    construct_models = [tuple(r) for r in iter_ipfs_cids]


    dollar_results_df = pd.DataFrame()
    for model_cid, ssle_cid, ds_features in construct_models:
        #Get Models
        clf = hh.get_model(model_cid)
        ss = hh.get_model(eval(ssle_cid)["ss"])
        le = hh.get_model(eval(ssle_cid)["le"])
        fc = eval(ds_features)["filter_cols"]
        stratify = eval(ds_features)["stratify"]
        smote = eval(ds_features)["smote"]

        #update this to always equal 100 and not be under or hover_data
        #good for right now though
        progress += int(100 / len(construct_models))

        p_bar.progress(progress)

        pred,acc,f1,precision,recall = prediction(clf,ss,le,df_predict,y,fc)

        dollar_preds,num_predictions = compare_strategy(st.session_state.ntp,tw,initial_amount,df_predict,y,ss,le,clf,fc)

        temp_df = br_cost_ds(dataset["Date"],str(clf).replace(" ",""),model_cid,
                    acc,dollar_preds["Strategy_br_cost"],pred)

        dollar_results_df = pd.concat([dollar_results_df,temp_df],axis=0)

    for scenario in ["Lowest","DAI","USDC","USDT"]:

        if scenario == "Lowest":
            classification = y

        else:
            classification_name = "_".join([scenario,"borrowRate"])

            classification = [classification_name] * len(y)

        df_fixed_strategy = br_cost_ds(dataset["Date"],scenario,"NA",0,
                    dollar_preds[f"{scenario}_br_cost"],classification
)

        dollar_results_df  = pd.concat([dollar_results_df,df_fixed_strategy],axis=0)

    d_colors = {'DAI_borrowRate':'gold',
              'USDT_borrowRate':'lightgreen',
              'USDC_borrowRate':'royalblue',
              }

    #offset to most recent prediction
    cummulative_df = (dollar_results_df.pivot(index="Date",columns="Strategy",
                        values="Borrowing Cost").cumsum()
                        .reset_index().iloc[(tw*3)*-1:]
)

    cummulative_df_formatted = (pd.melt(cummulative_df,id_vars=["Date"],
                                var_name="Strategy",value_name="Cumulative Borrow Cost")
)

    dollar_results_df_cmlt_plot = pd.merge(dollar_results_df,cummulative_df_formatted,on=["Date","Strategy"])

    fig = (px.scatter(dollar_results_df_cmlt_plot,x="Date",y="Cumulative Borrow Cost",
            color="Predictions",facet_col="Strategy",facet_col_wrap=4,hover_data = ["Borrowing Cost","Accuracy","Model_cid"],
            color_discrete_map=d_colors)
)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=",1)[1]))
    fig.update_layout(margin=dict(t=40),yaxis_title="Cummulative Borrowing Cost",
                        autosize=False,width=1200,height=700,legend=dict(font=dict(size= 20))
)
    fig.update_xaxes(showgrid=False)

    interest_cost_df = dollar_results_df.pivot(index=["Date"], columns="Strategy", values="Borrowing Cost")

    interest_cost_df = (pd.DataFrame(interest_cost_df.select_dtypes("float").sum())
                        .rename(columns={0:"Interest Cost"})
                        .reset_index()
)
    interest_cost_df["Interest Percentage"] = (interest_cost_df["Interest Cost"]/initial_amount)*100
    interest_cost_df["Est Annual Interest Percentage"] = interest_cost_df["Interest Percentage"] * 2

    interest_cost_df = (interest_cost_df.merge(dollar_results_df
                        .drop_duplicates("Strategy")[["Strategy","Accuracy"]],
                        on="Strategy")
                        )


    mapper =  {'Interest Cost': '${:,.0f}',
       'Interest Percentage': '{:,.2f}%',
       "Est Annual Interest Percentage": '{:,.2f}%',
       'Accuracy': '{:,.2f}%',
       }

    display_df = (interest_cost_df.set_index(["Strategy","Accuracy"])
                .sort_values("Interest Cost",ascending=True)
                .reset_index()
                .style.format(mapper)
                .highlight_min(color='green',subset=pd.IndexSlice[:, ["Interest Cost", "Interest Percentage"]])
)

    with chart2.expander("Parameters",expanded=True):
        st.dataframe(display_df)

    st.title("Historical Performance")
    current_prediction = pred[-1].split("_")[0]
    st.title(f"Best Model Chooses - {current_prediction} ")
    st.plotly_chart(fig,use_container_width=True)
    create_footer(dataset)


if __name__ == "__main__":
    main()
