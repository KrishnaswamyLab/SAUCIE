import numpy as np
import pandas as pd
import streamlit as st

from streamlit_elements.elements import display_buttons
from streamlit_elements.scores import display_scores
from streamlit_elements.elements import dump_model, convert_df
from streamlit_elements.figures import prepare_figure
from saucie.wrappers import SAUCIE_batches, SAUCIE_labels


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

        st.markdown("### Select original labels and batches")
        with st.form(key="my_form"):
            label_select = st.selectbox(
                "Label",
                options=["No labels"]+data.columns.tolist(),
                help="""
                Select which column refers to your labels.
                If none, select "No labels" and submit.
                """,
            )

            batch_select = st.selectbox(
                    "Batch",
                    options=["No batches"]+data.columns.tolist(),
                    help="""
                    Select which column refers to your batches.
                    If none, select "No batches" and submit.
                    """,
                )
            normalize = st.checkbox('Normalize my data', value=True)
            submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            if batch_select == "No batches":
                model_batches = None
            else:
                batches = data[[batch_select]].to_numpy()
                batches = batches.flatten()
                data.drop(columns=batch_select, inplace=True)
                model_batches = SAUCIE_batches(epochs=2, lr=1e-9,
                                               normalize=normalize,
                                               random_state=42)

            if label_select == "No labels":
                ground_truth = None
            else:
                ground_truth = data[[label_select]].to_numpy()
                ground_truth = ground_truth.flatten()
                data.drop(columns=label_select, inplace=True)

            data = data.to_numpy()
            data = data.astype(float)

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
            saucie = SAUCIE_labels(epochs=100, lr=1e-4, normalize=False,
                                   batch_size=256, shuffle=True)
            with st.spinner("Training the model"):
                saucie.fit(cleaned_data)
            embedded = saucie.transform(cleaned_data)
            labels = saucie.predict(cleaned_data)
            fig = prepare_figure(embedded[:, 0], embedded[:, 1],
                                 labels, ground_truth)

            st.plotly_chart(fig, use_container_width=True)

            display_scores(cleaned_data, embedded, labels, ground_truth)
            saucie_download = dump_model(saucie)
            labels_csv = convert_df(pd.DataFrame(labels))
            embedded_csv = convert_df(pd.DataFrame(embedded))
            if model_batches is not None:
                model_batches = dump_model(model_batches)
                cleaned_data = convert_df(pd.DataFrame(cleaned_data))
            # labels, embedding, model, cleaned data, model for batches
            display_buttons(labels_csv, embedded_csv, saucie_download,
                            cleaned_data, model_batches)
