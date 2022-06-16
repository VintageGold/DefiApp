import streamlit as st
import pandas as pd
from plotting import cumulative_interest,borrowing_rates
from prediction import prediction,compare_strategy
import plotly.express as px
import helpers.helper as hh

# set variables in session state
st.session_state.ntp = 5

#.cumsum()

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

def main():
    st.set_page_config(layout="wide",page_title="DefiSquad - Compound Model")
    st.image("https://uploads-ssl.webflow.com/61617acfb8ea62c4150005a1/61617ce3dd51f921e58fbd24_logo.svg", width=200)
    initial_amounts = sorted(range(int(10e6),int(10e7)+int(10e6),int(10e6)),reverse=True)
    title = st.title(f'Defi Borrowing Rate Saver :sunglasses: :fire:')
    chart1,chart2,chart3= st.columns((2,1,1))
    chart2.metric("Maximum Interest Saved", "1%", delta=None, delta_color="normal")
    chart2.metric("Maximum Dollars Saved", "$100,000", delta=None, delta_color="normal")

    with chart1.expander("Parameters",expanded=True):

        initial_amount = st.selectbox("Enter Loan Amount",
                                    options=initial_amounts,
                                    format_func=format_string)

        tw = st.selectbox("Time Window (Days)",options=[7,14,21])

    dollar_results = pd.DataFrame()

    response,log = hh.get_ipfs("bafkreicu7w25tzdmxeghhdnorfjgdv4apk66rnuy53gukjiv75c5smq6ha")

    dataset = hh.read_file(response.content)

    df_predict,y = hh.get_tabpandas_multi(dataset.drop(columns=["Date"]),st.session_state.ntp,tw)

    load_ipfs_cids = pd.read_csv(f"models/{st.session_state.ntp}_{tw}_models.csv")

    iter_ipfs_cids = (load_ipfs_cids[["ipfs_pin_hash","keyvalues_scalerLabelEncoder",
                        "keyvalues_datasetTraining"]]
                        .to_numpy().tolist()
    )

    for model_cid, ssle_cid, ds_features in [tuple(r) for r in iter_ipfs_cids]:

        clf = hh.get_model(model_cid)
        ss = hh.get_model(eval(ssle_cid)["ss"])
        le = hh.get_model(eval(ssle_cid)["le"])
        fc = eval(ds_features)["filter_cols"]

        pred,acc,f1,precision,recall = prediction(clf,ss,le,df_predict,y,fc)

        dollar_preds,num_predictions = compare_strategy(st.session_state.ntp,tw,initial_amount,df_predict,ss,le,clf,fc)

        dollar_results[str(clf).replace("\n","")] = dollar_preds["Strategy_br_cost"]


        dollar_results["Date"] = dataset[["Date"]][:len(dollar_results)]
        dollar_results["ntp_tw"] = "_".join([str(st.session_state.ntp),str(tw)])

    dollar_results["DAI_br_cost"] = dollar_preds["DAI_br_cost"]
    dollar_results["USDC_br_cost"] = dollar_preds["USDC_br_cost"]
    dollar_results["USDT_br_cost"] = dollar_preds["USDT_br_cost"]

    dollar_results_plot = pd.melt(dollar_results.copy(),id_vars=['ntp_tw','Date'],var_name=["strategy"],value_name="borrow_cost")

    chart3.metric("Number of Predictions",num_predictions, delta=None, delta_color="normal")
    chart3.metric("Number of Inputs",df_predict.shape[0], delta=None, delta_color="normal")
    # chart3.metric("Accuracy",f"{acc*100:,.0f}%", delta=None, delta_color="normal")

    fig = px.line(dollar_results_plot,x="Date",y="borrow_cost",color="strategy",title="Interest Payments")

    fig.update_layout(margin=dict(t=40))
    fig.update_xaxes(showgrid=False)

    mapper =  {'Interest Cost': '${:,.0f}',
       'Interest Percentage': '{:,.2f}%',
       }
    interest_cost_df = pd.DataFrame(dollar_results.select_dtypes("float").sum()).rename(columns={0:"Interest Cost"})
    interest_cost_df["Interest Percentage"] = (interest_cost_df["Interest Cost"]/initial_amount)*100
    display_df = (interest_cost_df
                .sort_values("Interest Cost",ascending=True)
                .style.format(mapper).highlight_min(color='green')
)


    st.dataframe(display_df)

    st.plotly_chart(fig,use_container_width=True)

#     dates = pd.date_range("Jan-01-2022","June-15-2022",freq="D")
#
#     df = pd.DataFrame({"value":list(range(0,dates.shape[0])),"date":list(dates)})
#     df["class"] = df["value"].apply(lambda x: "DAI_borrowRate"if x % 2 == 0 else "USDC_borrowRate")
#     fig = px.line(df,x="date",y="value",title="Interest Owed",color="class")
#     fig.update_layout(
#     margin=dict(t=40),
# )
#
#     st.plotly_chart(fig,use_container_width=True)
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
