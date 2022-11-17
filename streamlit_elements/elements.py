import io
import pickle

import streamlit as st


def display_buttons(labels, dim_red,
                    model, cleaned=None, model_batch=None):
    col1, col2 = st.columns([0.4, 1], gap="small")

    col1.download_button("Download labels ",
                         labels,
                         file_name="labels.csv",
                         mime='text/csv')

    col2.download_button("Download model",
                         model,
                         file_name="model.pickle",
                         help="""Download model for clustering
                          and dimensionality reduction""",
                         mime='application/octet-stream')

    col1.download_button("Download embedding",
                         dim_red,
                         file_name="reduced_data.csv",
                         mime='text/csv')

    if model_batch is not None:
        col2.download_button("Download model - batches",
                             model_batch,
                             file_name="model.csv",
                             help="""Download model for batch correction
                              and data cleaning""",
                             mime='text/csv')

        col1.download_button("Download cleaned data",
                             cleaned,
                             file_name="cleaned_data.csv",
                             mime='text/csv')


@st.cache
def dump_model(model):
    f = io.BytesIO()
    pickle.dump(model, f)
    f.seek(0)
    return f


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
