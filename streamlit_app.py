import numpy as np
import pandas as pd
import plotly_express as px
import streamlit as st

from streamlit_elements.elements import display_buttons, display_scores
from streamlit_elements.figures import prepare_figure
from saucie.wrappers import SAUCIE_batches, SAUCIE_labels


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


if __name__ == "__main__":
    st.set_page_config(
        page_title="SAUCIE", page_icon="🔎"
        )

    st.write("""
        # SAUCIE
        """)

    st.markdown("### Upload your data file")
    uploaded = st.file_uploader("your data file", type=[".csv"],
                                accept_multiple_files=False,
                                help="help here", on_change=None,
                                args=None, kwargs=None,
                                disabled=False, label_visibility="hidden")
    if uploaded:

        data = pd.read_csv(uploaded, index_col=0)
        df = px.data.iris()
        species = np.unique(np.array(df['species']))
        sp = np.array([species[i-1] for i in df['species_id'].tolist()])
        random_species = np.arange(0, 4)
        random_sp = np.random.randint(0, 4, size=sp.shape[0])
        y = df["sepal_length"].to_numpy()
        x = df["sepal_width"].to_numpy()
        st.markdown("### Select original labels and batches")
        with st.form(key="my_form"):
            label_select = st.selectbox(
                "Label",
                options=["No labels"]+data.index.tolist(),
                help="""
                Select which row refers to your labels.
                If none, select "No labels" and submit.
                """,
            )

            batch_select = st.selectbox(
                    "Batch",
                    options=["No batches"]+data.index.tolist(),
                    help="""
                    Select which row refers to your batches.
                    If none, select "No batches" and submit.
                    """,
                )
            normalize = st.checkbox('Normalize my data', value=True)
            submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            csv = convert_df(data)
            if batch_select == "No batches":
                model_batches = None
            else:
                batches = data.loc[batch_select].to_numpy()
                data.drop(index=batch_select)
                model_batches = SAUCIE_batches(epochs=2, lr=1e-9,
                                               normalize=normalize,
                                               random_state=42)

            if label_select == "No labels":
                ground_truth = None
            else:
                ground_truth = data.loc[label_select].to_numpy()
                data.drop(index=label_select)

            data = data.to_numpy().T

            if model_batches is not None:
                model_batches.fit(data, batches)
                # fit transform for batches
                cleaned_data = model_batches.transform(data, batches)
            else:
                if normalize:
                    data = (data - np.min(data, axis=0))
                    data = data/np.max(data, axis=0)
                    cleaned_data = np.arcsinh(data)
                else:
                    cleaned_data = data

            # fit on the cleaned data -> labels, embed
            saucie = SAUCIE_labels(epochs=50, lr=1e-5, normalize=False,
                                   batch_size=256, shuffle=True)
            saucie.fit(cleaned_data)
            encoded = saucie.transform(cleaned_data)
            labels = saucie.predict(cleaned_data)
            fig = prepare_figure(encoded[:, 0], encoded[:, 1],
                                 labels, ground_truth)

            st.plotly_chart(fig, use_container_width=True)
            vis = np.append(x.reshape(x.shape[0], -1),
                            y.reshape(y.shape[0], -1), axis=-1)
            # cleaned data or original, embedding, labels, ground truth
            # display_scores(cleaned_data, vis, random_sp, ground_truth)
            # labels, embedding, model, cleaned data, model for batches
            display_buttons(csv, csv, csv, cleaned_data, model_batches)
