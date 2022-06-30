import streamlit as st
import pandas as pd
from plotting import cumulative_interest,borrowing_rates
from prediction import prediction,compare_strategy
import plotly.express as px
import helpers.helper as hh

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

def ui_cols(size1,size2):

    return st.columns((size1,size1))

def main():
    st.set_page_config(layout="wide",page_title="DefiSquad - Compound Model")
    st.image("https://uploads-ssl.webflow.com/61617acfb8ea62c4150005a1/61617ce3dd51f921e58fbd24_logo.svg", width=200)
    st.image("https://cryptologos.cc/logos/compound-comp-logo.svg?v=022",width=100)
    title = st.title(f'Defi Borrowing Rate Saver :sunglasses: :fire:')
    st.write("https://defi.instadapp.io/compound")
    st.write("https://dune.com/datanut/Compound-Maker-and-Aave-Deposits-Loans-LTV")

    chart1,chart2= ui_cols(2,2)

    with chart1.expander("Interest Cost",expanded=True):

        initial_amounts = sorted(range(int(10e6),int(10e7)+int(10e6),int(10e6)),reverse=True)
        initial_amount = st.selectbox("Enter Loan Amount",
                                    options=initial_amounts,
                                    format_func=format_string)

        tw = st.selectbox("Time Window (Days)",options=[7,14,21])

        st.write("Model Loaded Progress")
        my_bar = st.progress(0)
        progress_amount = 0

    dollar_results_df = pd.DataFrame()

    #Get Data
    response,log = hh.get_ipfs("bafkreidnuulqq7cysvb35agjrn2st3nmhprjvp4eooew6z5xqyyigefav4")

    df_historical = hh.read_file(response.content)

    df_realtime = pd.read_csv("../data/compoundV2.csv")

    dataset = pd.concat([df_historical,df_realtime],axis=0).sort_values("Date",ascending=True)

    df_predict,y = hh.get_tabpandas_multi(dataset.drop(columns=["Date"]),st.session_state.ntp,tw)

    load_ipfs_cids = pd.read_csv(f"models/V99/{st.session_state.ntp}_{tw}_models.csv").iloc[:4]

    iter_ipfs_cids = (load_ipfs_cids[["ipfs_pin_hash","keyvalues_scalerLabelEncoder",
                        "keyvalues_datasetTraining"]]
                        .to_numpy().tolist()
    )

    construct_models = [tuple(r) for r in iter_ipfs_cids]

    for model_cid, ssle_cid, ds_features in construct_models:
        #Get Models
        clf = hh.get_model(model_cid)
        ss = hh.get_model(eval(ssle_cid)["ss"])
        le = hh.get_model(eval(ssle_cid)["le"])
        fc = eval(ds_features)["filter_cols"]

        #update this to always equal 100 and not be under or hover_data
        #good for right now though
        progress_amount += int(100 / len(construct_models))

        my_bar.progress(progress_amount)

        pred,acc,f1,precision,recall = prediction(clf,ss,le,df_predict,y,fc)

        dollar_preds,num_predictions = compare_strategy(st.session_state.ntp,tw,initial_amount,df_predict,y,ss,le,clf,fc)
        temp_df = br_cost_ds(dataset["Date"],str(clf).replace(" ",""),model_cid,
                    acc,dollar_preds["Strategy_br_cost"],pred
)

        dollar_results_df = pd.concat([dollar_results_df,temp_df],axis=0)

    for scenario in ["Lowest","DAI","USDC","USDT"]:

        if scenario == "Lowest":
            classification = y

        else:
            classification_name = "_".join([scenario,"borrowRate"])

            classification = [classification_name] * len(y)

        df_fixed_strategy = br_cost_ds(dataset["Date"],scenario,"NA",
                    0,dollar_preds[f"{scenario}_br_cost"],classification)

        dollar_results_df  = pd.concat([dollar_results_df,df_fixed_strategy],axis=0)

    d_colors = {'DAI_borrowRate':'gold',
              'USDT_borrowRate':'lightgreen',
              'USDC_borrowRate':'royalblue',
              }

    cummulative_df = (dollar_results_df.pivot(index="Date",columns="Strategy",
                        values="Borrowing Cost").cumsum()
                        .reset_index()
)

    cummulative_df_formatted = (pd.melt(cummulative_df,id_vars=["Date"],
                                var_name="Strategy",value_name="Cumulative Borrow Cost")
)

    dollar_results_df_cmlt_plot = pd.merge(dollar_results_df,cummulative_df_formatted,on=["Date","Strategy"])

    fig = (px.scatter(dollar_results_df_cmlt_plot,x="Date",y="Cumulative Borrow Cost",
            color="Predictions",facet_col="Strategy",facet_col_wrap=4,hover_data = ["Borrowing Cost","Accuracy","Model_cid"],
            title="Interest Payments",color_discrete_map=d_colors)
)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=",1)[1]))
    fig.update_layout(margin=dict(t=40),yaxis_title="Cummulative Borrowing Cost")
    fig.update_xaxes(showgrid=False)

    interest_cost_df = dollar_results_df.pivot(index=["Date"], columns="Strategy", values="Borrowing Cost")

    interest_cost_df = (pd.DataFrame(interest_cost_df.select_dtypes("float").sum())
                        .rename(columns={0:"Interest Cost"})
                        .reset_index()
)
    interest_cost_df["Interest Percentage"] = (interest_cost_df["Interest Cost"]/initial_amount)*100

    interest_cost_df = (interest_cost_df.merge(dollar_results_df
                        .drop_duplicates("Strategy")[["Strategy","Accuracy"]],
                        on="Strategy")
                        )


    mapper =  {'Interest Cost': '${:,.0f}',
       'Interest Percentage': '{:,.3f}%',
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
    st.metric("Compund Borrow Position",f"Best Model Chooses - {current_prediction} ")
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(dataset)

    title = st.title('Model Description')
    st.write("""
    ```
    Stable Tokens -  DAI, USDC, USDT \n
    Strategy -  Predict Token with lowest mean/average borrowing rate over TW \n
    Description \n
        In the example below there are the choice between borrowing DAI, USDC, USDT. \n
        Our model looks at the past 5 days to predict which stable coin will have the lowest mean/average interest rate over the next 7, 14, or 21 days.\n
        In the example below DAI was chosen.
    ```
    """
    )


    st.image("https://ipfs.io/ipfs/bafkreidqulmh3jp2oeea2fth7q3ilx3y456magfqhnj22zmz6gnz4ijrhy")

    st.write((
            "bafkreifiadt2yrvutdehggmbs44go5zwznbhgz3h5tqixc7m4lw3qinrxa",
            "bafkreih4xcizggjqyocg5g5mzqkflalpymuprbfsowj2j7uyvvsmjx535q",
            "bafkreifujjs4zark5oyuvfgyobjcnym4ftvjosvlgc2vu73f6n25ky72xe",
            "bafkreibzrxyzz6koxy2jrdbhpnwctdyrjttgkjbbhshmdjopcbbrfnmovu",
            "bafkreia4m7wo5kbjsy6qe2jlnouayh3ayzb32lvyrckrlcfzhlydchmwwa",
            "bafkreihjukozyctyfstuqzlphmpllwjiiotbs2gpitswtiee646xwwigry",
            "bafkreid2s7gdmch6m7qhewazbn6bv5m4qulw6lohelkz42uobtdgsuppxu",
            "Data: QmWAQjxm6CKaAjHjwPSqWmN8RMecuHS1bxEufqCNVSq96e",
            "Scaler: bafkreibr7ffetwyev6dumwnwxnzfcqjs7h3m5nmwekvvrkdhwwba6swudq",
            "Label: bafkreifzdn6zrrste3b7uquuwgonc63padomw2bdbjod4wbgewet4v336u"
            ))
#     #Line Plot
#     df_complete = pd.merge(chart1_pred_cost,chart2_pred_cost[["Date","Classification","M2_br_cost"]],on="Date")
#
#     df_complete["Classification_x"] = df_complete["Classification_x"].apply(lambda x: " ".join(["M1",x]))
#     df_complete["Classification_y"] = df_complete["Classification_y"].apply(lambda y: " ".join(["M2",y]))
#
#     #DataFrame
#     df_total_interest = (pd.DataFrame(df_complete.select_dtypes(include="float").sum(),
#                         columns=["Interest Owed"]
#                         )
#                         .sort_values("Interest Owed",ascending=False)
#                         )
#
#     df_total_interest["Interest Percentage"] = (df_total_interest["Interest Owed"]/initial_amount)*100
#
#
#     mapper =  {'Interest Owed': '${:,.0f}',
#        'Interest Percentage': '{:,.2f}%',
#        }
#
#     display_df = df_total_interest.style.format(mapper).highlight_min(color='green')
#
#     interest_saved = df_total_interest["Interest Percentage"].max() - df_total_interest["Interest Percentage"].min()
#
#     st.metric("Savings",interest_saved, delta=None, delta_color="normal")
#
#     with chart1.expander("Total Interest Cost",expanded=True):
#
#
#         st.dataframe(display_df)
#
#         # display_interest_df = df_total_interest[["Interest Percentage"]].style.format("{:,.2f}%").highlight_min(color='green')
#         #
#         #
#         # display_dollar_df = df_total_interest.style.format("${:,.0f}").highlight_min(color='green')
#
#     # with chart2.expander(f"Total Interest Cost Percentage, based on ${initial_amount:,.0f}",expanded=True):
#
#
#     fig = cumulative_interest(df_complete,"Date","M1_br_cost",
#                         "M2_br_cost","Classification_x","Classification_y")
#
#     st.plotly_chart(fig,use_container_width=True)
#
#
#
#     #Line Plot
#     # fig2 = borrowing_rates(df_complete,"Date","M1_br_cost","M2_br_cost")
#     #
#     # st.plotly_chart(fig2,use_container_width=True)
#     #
#     # with sidebar:
#     #     st.header("Tables")
#     #
#     #     st.write("Total Interest Expense Savings")
#     #     df_total_interest = (pd.DataFrame(df_complete.select_dtypes(include="float").sum(),
#     #                         columns=["Interest Owed"]
#     #                         )
#     #                         .sort_values("Interest Owed",ascending=False)
#     #                         )
#     #     display_df = df_total_interest.style.format("${:,.0f}").highlight_min(color='green')
#     #
#     #     st.dataframe(display_df)
#     #
#     #     st.write("Cummulative Interest Rates")
#     #     st.dataframe(df_complete.select_dtypes(include="float").cumsum().style.format("${:,.0f}"))
#     #
#     #     st.write("Interest Rates")
#     #     st.dataframe(df_complete)
#
if __name__ == "__main__":
    main()
