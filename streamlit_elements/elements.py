import streamlit as st
from sklearn.metrics import adjusted_rand_score, silhouette_score


def display_buttons(labels, dim_red,
                    model, cleaned=None, model_batch=None):
    col1, col2 = st.columns([0.4, 1], gap="small")

    col1.download_button("Download labels ",
                         labels,
                         file_name="labels.csv",
                         mime='text/csv')

    col2.download_button("Download model",
                         model,
                         file_name="model.csv",
                         help="""Download model for clustering
                          and dimensionality reduction""",
                         mime='text/csv')

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
def compute_ari_score(y_pred, y_true=None):
    ari_score = adjusted_rand_score(y_true, y_pred)
    return ari_score


@st.cache
def compute_silhouette(x, y):
    score = silhouette_score(x, y)
    return score


def display_scores(X, X_embed, y_pred, y_true=None):
    col1, col2 = st.columns(2)
    silhouette_labels = compute_silhouette(X, y_pred)
    col1.metric("Silhouette of the clustering",
                value=round(silhouette_labels, 5))
    if y_true is not None:
        ari = compute_ari_score(y_pred, y_true)
        silhouette_embed = compute_silhouette(X_embed, y_true)
        col2.metric("Adjusted Rand Index",
                    value=round(ari, 5))
        col1.metric("Silhouette of the embedding",
                    value=round(silhouette_embed, 5))
