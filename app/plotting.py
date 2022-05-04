import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio

def cumulative_interest(df,x,y1,y2,color1,color2):

    # assign colors to type using a dictionary
    colors = {'DAI_br_cost':'gold',
              'USDT_br_cost':'green',
              'USDC_br_cost':'blue',
              }


    # plotly figure
    fig1=go.Figure()
    for t in df[colors.keys()]:
        fig1.add_traces(go.Scatter(x=df[x], y=df[t].cumsum(), name=t,
                             marker_color=colors[t],
                             mode="markers+lines"))

    M1_cs = df[y1].cumsum()

    fig2 = px.scatter(df,x=x,y=M1_cs,color=color1,template="plotly_dark")

    # st.plotly_chart(fig,use_container_width=True)

    M2_cs = df[y2].cumsum()

    fig3 = px.scatter(df,x=x,y=M2_cs,color=color2,template="plotly_dark")

    # st.plotly_chart(fig,use_container_width=True)

    fig = go.Figure(data = fig1.data + fig2.data + fig3.data)
    fig.update_layout(
    title="Cumulative Interest Cost",
    plot_bgcolor="black"
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink',title_text="Date")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink',title_text="Dollars")

    return fig

def borrowing_rates(df,x,y1,y2):
    # assign colors to type using a dictionary
    colors = {'DAI_br_cost':'gold',
              'USDT_br_cost':'green',
              'USDC_br_cost':'blue',
              }
    # # plotly figure
    fig1=go.Figure()
    for t in df[colors.keys()]:
        fig1.add_traces(go.Scatter(x=df[x], y=df[t], name=t,
                             marker_color=colors[t],
                             mode="markers")
                        )

    df_temp = df.copy()
    df_temp["color"] = "Strategy"

    fig2 = px.line(df_temp,x=x,y=y1,template="plotly_dark",
                    color_discrete_sequence=['red'])

    fig3 = px.line(df_temp,x=x,y=y2,template="plotly_dark",
                    color_discrete_sequence=['white'])

    fig = go.Figure(data = fig1.data + fig2.data + fig3.data)
    fig.update_layout(
    title="Borrowing Rates Overtime",
    plot_bgcolor="black"
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink',title_text="Date")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink',title_text="Dollars")

    return fig
