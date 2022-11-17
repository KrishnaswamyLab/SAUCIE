import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


@st.cache
def prepare_figure(x, y, saucie_label, ground_truth=None):
    if ground_truth is not None:
        fig = prepare_subplots()
        fig = get_subplot(fig, x, y, ground_truth,
                          title="Original labels", col=1)
        fig = get_subplot(fig, x, y, saucie_label,
                          title="SAUCIE labels", col=2)
    else:
        fig = prepare_single_plot()
        fig = add_to_single_plot(fig, x, y, saucie_label)
    return fig


def prepare_subplots():
    fig = make_subplots(rows=1, cols=2,
                        shared_xaxes=True,
                        subplot_titles=("Original labels",
                                        "SAUCIE results"),
                        shared_yaxes=True)
    fig.update_xaxes(title_text='SAUCIE 1')
    fig.update_yaxes(title_text='SAUCIE 2')
    fig.update_xaxes(matches='x')
    fig.update_layout(legend=dict(groupclick="toggleitem"))
    return fig


def prepare_single_plot():
    fig = go.Figure()
    fig.update_layout(title_text="SAUCIE results")
    fig.update_xaxes(title_text='SAUCIE 1')
    fig.update_yaxes(title_text='SAUCIE 2')
    return fig


def add_to_single_plot(fig, x, y, labels):
    single_labels = np.unique(labels)
    for label in single_labels:
        fig.add_trace(go.Scatter(x=x[np.where(labels == label)],
                                 name=str(label),
                                 showlegend=True,
                                 y=y[np.where(labels == label)],
                                 mode='markers'))
    return fig


def get_subplot(fig, x, y, labels, title="", col=1):
    single_labels = np.unique(labels)
    for label in single_labels:
        fig.add_trace(go.Scatter(x=x[np.where(labels == label)],
                                 name=str(label),
                                 showlegend=True,
                                 legendgroup=str(col),
                                 legendgrouptitle_text=title,
                                 y=y[np.where(labels == label)],
                                 mode='markers'),
                      row=1, col=col)
    return fig
