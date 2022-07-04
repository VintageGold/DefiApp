import streamlit as st

def create_header():
    st.set_page_config(layout="wide",page_title="DefiSquad - Compound Model")
    st.image("https://uploads-ssl.webflow.com/61617acfb8ea62c4150005a1/61617ce3dd51f921e58fbd24_logo.svg", width=200)
    st.image("https://cryptologos.cc/logos/compound-comp-logo.svg?v=022",width=100)
    title = st.title(f'Defi Borrowing Rate Saver :sunglasses: :fire:')
    st.write("https://defi.instadapp.io/compound")
    st.write("https://dune.com/hagaetc/lending")


def create_footer(df):
    st.title('Model Description')
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

    st.image("https://ipfs.io/ipfs/bafybeiayn5tfdsku32crrj4qro6k5evuyp57epfjkin7it6jn4q7xqjywi")

    st.title("Dataset")
    st.dataframe(df)


def create_progressbar():
    p_bar = st.progress(0)
    progress = 0

    return p_bar,progress
