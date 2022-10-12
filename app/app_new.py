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
st.session_state.start_date = "2021-12-27"
st.session_state.model_count = 4

def get_metadata(model_cid,ssle_cid,ds_features):
    #Get Models
    clf = hh.get_model(model_cid)
    ss = hh.get_model(eval(ssle_cid)["ss"])
    le = hh.get_model(eval(ssle_cid)["le"])
    fc = eval(ds_features)["filter_cols"]
    stratify = eval(ds_features)["stratify"]
    smote = eval(ds_features)["smote"]

    return clf,ss,le,fc,stratify,smote

def format_string(initial_amount):

    return f"${initial_amount:,}"

@st.cache(persist=True)
def load_dataset():
    #Old - bafybeifd7qozy2aw5anuk6v6p6jcjva3pwnotptrfnj2zsukgeynmo5mky
    response = requests.get("https://ipfs.io/ipfs/bafybeig7pplmikh3w43d2bp7r4lc24nspjljrjljutlsghvj6qnglxnxku")

    df = hh.read_file(response.content).sort_values("Date",ascending=True).drop(columns=["timestamp"])

    return df.query(f'Date >= "{st.session_state.start_date}"')


def next_dates_to_checkin(df,td,num=4):

    pred = td*num

    dates = list()

    for i in range(0,pred,td):

        checkindate = df["Date"].max() + timedelta(days=i)
        dates.append(str(checkindate).split(" ")[0])

    return dates


def main(debug):
    create_header()

    chart1,chart2= st.columns((2,2))

    with chart1.expander("General",expanded=True):

        initial_amounts = sorted(range(int(10e6),int(10e7)+int(10e6),int(10e6)),reverse=False)
        initial_amount = st.selectbox("Enter Loan Amount",
                                    options=initial_amounts,
                                    format_func=format_string)

        tw = st.selectbox("Time Window (Days)",options=[7,14,21])

        st.write("Model Load Progress")
        p_bar,progress = create_progressbar()

    #Get Data

    dataset = load_dataset()

    df_predict,y = hh.get_tabpandas_multi(dataset.drop(columns=["Date"]),st.session_state.ntp,tw)

    load_ipfs_cids = pd.read_csv(f"models/V1/{st.session_state.ntp}_{tw}_models.csv").iloc[:st.session_state.model_count]

    iter_ipfs_cids = (load_ipfs_cids[["ipfs_pin_hash","keyvalues_scalerLabelEncoder",
                        "keyvalues_datasetTraining"]]
                        .to_numpy().tolist()
        )

    construct_models = [tuple(r) for r in iter_ipfs_cids]

    dollar_results_df = pd.DataFrame()

    col_mapping = dict()
    metrics_mapping = dict()
    i = 0

    for model_cid, ssle_cid, ds_features in construct_models:

        clf,ss,le,fc,stratify,smote = get_metadata(model_cid, ssle_cid, ds_features)

        model_name = str(clf).replace(" ","")

        pred,acc,f1,precision,recall = prediction(clf,ss,le,df_predict,y,fc)

        dollar_preds,chunks = compare_strategy(st.session_state.ntp,tw,initial_amount,df_predict,y,ss,le,clf,fc)

        col_mapping[model_name] = "".join(["M",str(i)])
        metrics_mapping["".join(["M",str(i)])] = acc

        dollar_results_df = pd.concat([dollar_results_df,dollar_preds],axis=1)

        progress += int(100 / len(construct_models))

        p_bar.progress(progress)

        i+=1
    
    stable_tw_df = (df_predict[['DAI_borrowRate_t-0','USDC_borrowRate_t-0','USDT_borrowRate_t-0']]
                    .rename(columns={"DAI_borrowRate_t-0":"DAI_rate",
                      'USDC_borrowRate_t-0':'USDC_rate',
                      'USDT_borrowRate_t-0':'USDT_rate'})
)
    stable_tw_df["Lowest_rate"] = stable_tw_df.min(axis=1)
    stable_tw_df["Lowest_pred"] = stable_tw_df.idxmin(axis=1)
    stable_tw_df["DAI_pred"] = "DAI_borrowRate"
    stable_tw_df["USDC_pred"] = "USDC_borrowRate"
    stable_tw_df["USDT_pred"] = "USDT_borrowRate"

    dollar_results_df = (dollar_results_df.copy().reset_index()
                        .merge(stable_tw_df.copy(),left_index=True,right_index=True)
                        .merge(dataset.reset_index()[["Date"]],left_index=True,right_index=True)
                        .iloc[st.session_state.ntp:]
    )


    dollar_results_df_amt = (dollar_results_df.copy()[[col for col in dollar_results_df if col.endswith("_rate")]]
                /365*initial_amount 
                )

    dollar_results_df_amt.columns = [col.replace("_rate","_amt") for col in dollar_results_df_amt]

    amt_rate_dollar_results_df = dollar_results_df.merge(dollar_results_df_amt,left_index=True,right_index=True)


    cumsum_df = (amt_rate_dollar_results_df
    [[col for col in amt_rate_dollar_results_df.columns if col.endswith("_amt")]]
    .cumsum()
    )

   

    df_cummul = cumsum_df.merge(amt_rate_dollar_results_df
                    [[col for col in amt_rate_dollar_results_df if not col.endswith("_amt")]],
                    left_index=True,
                    right_index=True
    ).drop(columns="index").set_index("Date")


    amt_melt = df_cummul[[col for col in df_cummul if col.endswith("_amt")]].reset_index().melt(id_vars="Date",var_name="Model",value_name="Amount")

    amt_melt["Model"] = amt_melt["Model"].str.replace("_amt","")

    pred_melt = df_cummul[[col for col in df_cummul if col.endswith("_pred")]].reset_index().melt(id_vars="Date",var_name="Model",value_name="Pred")

    pred_melt["Model"] = pred_melt["Model"].str.replace("_pred","")

    df_plot = amt_melt.merge(pred_melt,on=["Date","Model"])


    df_plot[["Pred","_"]] = df_plot["Pred"].str.split("_",1,expand=True)

    #### Chart 2
    mapper =  {'Interest Cost': '${:,.0f}',
    'Avg Interest Percentage': '{:,.2f}%',
    "Est Annual Interest Percentage": '{:,.2f}%',
    'Accuracy': '{:,.2f}%',
    }

    with chart2.expander("Interest Cost",expanded=True):

        df_interest = (df_cummul
                    [[col for col in df_cummul.columns if col.endswith("_amt")]]
                    .tail(1)     
                    .T
        )

        df_interest.index = df_interest.index.str.replace("_amt","")

        df_interest = df_interest.rename(index=col_mapping)

        df_interest = (df_interest
        .rename(columns={df_interest.columns[0]:"Interest Cost"})
        .sort_values("Interest Cost",ascending=True)
        )

        df_interest["Avg Interest Percentage"] =  (df_interest["Interest Cost"] / initial_amount)*100

        def mapping(item,mapping_table):
            
            if item in mapping_table:
                return mapping_table[item]*100

            else:
                return 0

        
        
        df_interest["Accuracy"] = [mapping(i,metrics_mapping) for i in df_interest.index]

        df_interest_view = (df_interest
        .style.format(mapper)
        .highlight_min(color='green',subset=pd.IndexSlice[:, ["Interest Cost", "Avg Interest Percentage"]])

        )

        st.dataframe(df_interest_view)

    marker_colors = {'DAI':'gold',
            'USDT':'lightgreen',
            'USDC':'royalblue',
            }

    fig = px.scatter(df_plot,x="Date",y="Amount",color="Pred",color_discrete_map=marker_colors,
                     facet_col="Model",facet_col_wrap=4,title="Cummulative Amount")

    fig.update_layout(margin=dict(t=40),yaxis_title="Cummulative Borrowing Cost",
                        autosize=False,width=1200,height=700,legend=dict(font=dict(size= 20)),
    )
    fig.update_xaxes(showgrid=False)

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=",1)[1].split("_",1)[0]))

    fig.for_each_xaxis(lambda axis: axis.title.update(font=dict(size=20)))
    fig.for_each_yaxis(lambda axis: axis.title.update(font=dict(size=20)))
    st.plotly_chart(fig,use_container_width=True)
    
    create_footer(df_predict)

    if debug:
        st.title("Debugging")
        st.write("Org Dataset")
        st.dataframe(df_predict)
        st.write("Mappings")
        st.write(col_mapping)
        st.write(metrics_mapping)
        st.write("Raw DF")
        st.dataframe(dataset)
        st.write("Dollar result")
        st.dataframe(dollar_preds)
        st.dataframe(dollar_results_df)
        st.dataframe(amt_rate_dollar_results_df)
        st.write("Stable coin DF")
        st.dataframe(stable_tw_df)
        st.write("Cummulative DF")
        st.dataframe(cumsum_df)
        st.dataframe(df_cummul)
        st.write("Melt")
        st.dataframe(amt_melt)
        st.dataframe(pred_melt)
        st.write("Plotted DF")
        st.dataframe(df_plot)
    


#     interest_cost_df = (pd.DataFrame(dollar_results_df_d.rename(columns=mapping).select_dtypes("float").sum())
#                         .reset_index()
#                         .rename(columns={0:"Interest Cost","index":"Models"})

#     )

#     interest_cost_df["Models"] = interest_cost_df["Models"].apply(lambda x: x.split("_")[0])
#     interest_cost_df["Accuracy"] = interest_cost_df["Models"].apply(lambda x: metrics_mapping[x]*100 if x in metrics_mapping else 0)

#     interest_cost_df["Interest Percentage"] = (interest_cost_df["Interest Cost"]/initial_amount)*100
#     interest_cost_df["Est Annual Interest Percentage"] = interest_cost_df["Interest Percentage"] * 2

#     mapper =  {'Interest Cost': '${:,.0f}',
#        'Interest Percentage': '{:,.2f}%',
#        "Est Annual Interest Percentage": '{:,.2f}%',
#        'Accuracy': '{:,.2f}%',
#        }

#     display_df = (interest_cost_df.set_index(["Models","Accuracy"]).reset_index().set_index(["Models"])
#             .sort_values("Interest Cost",ascending=True)
#             .style.format(mapper)
#             .highlight_min(color='green',subset=pd.IndexSlice[:, ["Interest Cost", "Interest Percentage"]])
# )

#     with chart2.expander("Parameters",expanded=True):
#         st.dataframe(display_df)
#     current_prediction = pred[-1].split("_")[0]
#     st.header(f"Prediction for {str(df_cummulative_roworiented['Date'].max()).split(' ')[0]}")
#     st.metric("M0", "USDC")
#     st.metric("M1", "USDC")
#     st.metric("M2", "USDC")
#     st.metric("M3", "USDC")
#     st.write(next_dates_to_checkin(w,tw))
#     st.header("Historical Performance")



#     st.plotly_chart(fig,use_container_width=True)
#     create_footer(dataset.set_index(["Date"]))


# fig.for_each_annotation(lambda a: a.update(text=a.text.split("=",1)[1]))
if __name__ == "__main__":
    main(debug=True)
