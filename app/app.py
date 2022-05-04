import streamlit as st
import pandas as pd
from data.load import get_data,get_model
from ensemble_sklearn import SklearnEnsemble
from plotting import cumulative_interest,borrowing_rates

# set variables in session state
st.session_state.ntp = 5
st.session_state.tw = 7

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
    st.set_page_config(layout="wide",page_title="DeepDefi - Compound Model")
    st.image("https://uploads-ssl.webflow.com/61617acfb8ea62c4150005a1/61617ce3dd51f921e58fbd24_logo.svg", width=200)
    initial_amounts = sorted(range(int(10e6),int(10e7)+int(10e6),int(10e6)),reverse=True)

    st.write("""
    Stable Tokens: DAI, USDC, USDT \n
    Strategy: Predict Token with lowest mean borrowing rate over TW \n
    Use the CIDs below to compare Models at NTP 5 days and 7 days TW: \n
    """)
    st.write((
            "bafkreifiadt2yrvutdehggmbs44go5zwznbhgz3h5tqixc7m4lw3qinrxa",
            "bafkreih4xcizggjqyocg5g5mzqkflalpymuprbfsowj2j7uyvvsmjx535q",
            "bafkreifujjs4zark5oyuvfgyobjcnym4ftvjosvlgc2vu73f6n25ky72xe",
            "bafkreibzrxyzz6koxy2jrdbhpnwctdyrjttgkjbbhshmdjopcbbrfnmovu",
            "bafkreia4m7wo5kbjsy6qe2jlnouayh3ayzb32lvyrckrlcfzhlydchmwwa",
            "bafkreihjukozyctyfstuqzlphmpllwjiiotbs2gpitswtiee646xwwigry",
            "bafkreid2s7gdmch6m7qhewazbn6bv5m4qulw6lohelkz42uobtdgsuppxu",

            ))

    chart1,chart2,chart3= st.columns((2,2,2))
    sidebar = st.sidebar

    with sidebar:
        title = st.title(f'Defi Borrowing Rate Prediction')
        st.write(f"Number of Timepoints: {st.session_state.ntp}")
        st.write(f"Time Window: {st.session_state.tw}")
        initial_amount = st.selectbox("Enter Loan Amount",
                                    options=initial_amounts,
                                    format_func=format_string)

        data_cid_tb =  st.text_input("Enter Data IPFS CID",
                            key="data_cid",
                            value="QmWAQjxm6CKaAjHjwPSqWmN8RMecuHS1bxEufqCNVSq96e",
                            )

        df = get_data(data_cid_tb)

        scaler_cid_tb =  st.text_input("Enter IPFS CID for Scaler",
                                key="scaler_cid",
                                value="bafkreidhiiky34zsc7iqloh75izidtrk5ba4paavvuizdt4osshnrvvmou",
                                )

        scaler = get_model(scaler_cid_tb)
        st.write(scaler)

        le_cid_tb =  st.text_input("Enter IPFS CID for Label Encoder",
                                    key="label_cid",
                                    value="bafkreifzdn6zrrste3b7uquuwgonc63padomw2bdbjod4wbgewet4v336u",
                       )

        Le = get_model(le_cid_tb)
        st.write(Le)

    # with chart1.expander("Predictive Model 1",expanded=True):

        chart1_model_cid_tb =  st.text_input("Enter Model 1 IPFS CID", key="model_cid_chart3",
                                    value="bafkreih4xcizggjqyocg5g5mzqkflalpymuprbfsowj2j7uyvvsmjx535q",
                                    )
        chart1_model = get_model(chart1_model_cid_tb)
        st.write(chart1_model)
        chart1_pred_cost = predict(int(initial_amount),chart1_model,
                scaler,Le,df,
                st.session_state.ntp,st.session_state.tw,
                only_model=False
                )

    # with chart2.expander("Predictive Model 2",expanded=True):

        chart2_model_cid_tb =  st.text_input("Enter Model 2 IPFS CID", key="model_cid_chart4",
                                    value="bafkreifiadt2yrvutdehggmbs44go5zwznbhgz3h5tqixc7m4lw3qinrxa",
                                    )
        chart2_model = get_model(chart2_model_cid_tb)
        st.write(chart2_model)
        chart2_pred_cost = predict(int(initial_amount),chart2_model,
                scaler,Le,df,
                st.session_state.ntp,st.session_state.tw,
                only_model=True
                )

    #Line Plot
    df_complete = pd.merge(chart1_pred_cost,chart2_pred_cost[["Date","Classification","M2_br_cost"]],on="Date")

    df_complete["Classification_x"] = df_complete["Classification_x"].apply(lambda x: " ".join(["M1",x]))
    df_complete["Classification_y"] = df_complete["Classification_y"].apply(lambda y: " ".join(["M2",y]))


    with chart1.expander("Total Interest Expense Cost",expanded=True):
        #DataFrame
        df_total_interest = (pd.DataFrame(df_complete.select_dtypes(include="float").sum(),
                            columns=["Interest Owed"]
                            )
                            .sort_values("Interest Owed",ascending=False)
                            )


        display_df = df_total_interest.style.format("${:,.0f}").highlight_min(color='green')

        st.dataframe(display_df)

    with chart2.expander(f"Total Interest Percentage, based on ${initial_amount:,.0f}",expanded=True):

        df_total_interest["Interest Percentage"] = (df_total_interest["Interest Owed"]/initial_amount)*100

        display_df = df_total_interest[["Interest Percentage"]].style.format("{:,.2f}%").highlight_min(color='green')

        st.dataframe(display_df)

    fig = cumulative_interest(df_complete,"Date","M1_br_cost",
                        "M2_br_cost","Classification_x","Classification_y")

    st.plotly_chart(fig,use_container_width=True)



    #Line Plot
    # fig2 = borrowing_rates(df_complete,"Date","M1_br_cost","M2_br_cost")
    #
    # st.plotly_chart(fig2,use_container_width=True)
    #
    # with sidebar:
    #     st.header("Tables")
    #
    #     st.write("Total Interest Expense Savings")
    #     df_total_interest = (pd.DataFrame(df_complete.select_dtypes(include="float").sum(),
    #                         columns=["Interest Owed"]
    #                         )
    #                         .sort_values("Interest Owed",ascending=False)
    #                         )
    #     display_df = df_total_interest.style.format("${:,.0f}").highlight_min(color='green')
    #
    #     st.dataframe(display_df)
    #
    #     st.write("Cummulative Interest Rates")
    #     st.dataframe(df_complete.select_dtypes(include="float").cumsum().style.format("${:,.0f}"))
    #
    #     st.write("Interest Rates")
    #     st.dataframe(df_complete)

if __name__ == "__main__":
    main()
