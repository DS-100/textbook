"""
Imports and functions that all textbook pages load by default.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython import get_ipython
from IPython.display import display, set_matplotlib_formats, HTML
import myst_nb

import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# set up plotly defaults
pio.renderers.default = "plotly_mimetype+svg"
pio.templates["book"] = go.layout.Template(
    layout=dict(
        margin=dict(l=10, r=10, t=10, b=10),
        autosize=True,
        width=350,
        height=250,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+book"

# set up matplotlib defaults
get_ipython().run_line_magic("matplotlib", "inline")
set_matplotlib_formats("svg")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (4, 3)

# display options for numpy and pandas
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.set_option("display.max_rows", 7)
pd.set_option("display.max_columns", 8)
pd.set_option("precision", 2)
# stops scientific notation for pandas
# pd.set_option('display.float_format', '{:.2f}'.format)


def display_df(
    df, rows=pd.options.display.max_rows, cols=pd.options.display.max_columns
):
    """Displays n rows and cols from df"""
    with pd.option_context("display.max_rows", rows, "display.max_columns", cols):
        display(df)


def dfs_side_by_side(*dfs):
    """Displays two or more dataframes side by side"""
    display(
        HTML(
            f"""
        <div style="display: flex; gap: 1rem;">
        {''.join(df.to_html() for df in dfs)}
        </div>
    """
        )
    )


def df_interact(df, nrows=7, ncols=7):
    """
    Outputs sliders that show rows and columns of df
    """

    def peek(row=0, col=0):
        return df.iloc[row : row + nrows, col : col + ncols]

    if len(df.columns) <= ncols:
        interact(peek, row=(0, len(df), nrows), col=fixed(0))
    else:
        interact(peek, row=(0, len(df), nrows), col=(0, len(df.columns) - ncols))
    print("({} rows, {} columns) total".format(df.shape[0], df.shape[1]))


##############################################################################
# Plotly
##############################################################################

# When two traces share bingroup, plotly thinks they're the same plot
_clear = ["bingroup"]


def _clear_prop(trace, prop):
    if hasattr(trace, prop):
        trace.update({prop: None})


def _clear_props(traces):
    for trace in traces:
        for prop in _clear:
            _clear_prop(trace, prop)


def plots_in_row(figures, width=700, height=250, **kwargs):
    """Combine plotly figures side by side"""
    fig = make_subplots(cols=len(figures), **kwargs)
    fig.update_layout(width=width, height=height)

    traces = [next(fig.select_traces()) for fig in figures]
    _clear_props(traces)
    for i, trace in enumerate(traces):
        fig.add_trace(trace, row=1, col=i + 1)
    return fig


def left_right(left, right, width=700, height=250, **kwargs):
    """Two plotly figures side by side"""
    return plots_in_row([left, right], width=width, height=height, **kwargs)


def margin(fig, **kwargs):
    """Set margins for a plotly figure"""
    return fig.update_layout(margin=kwargs)


def title(fig, label, **kwargs):
    """Set title for a plotly figure"""
    return fig.update_layout(
        title={
            "text": label,
            **kwargs,
        }
    )


def xlabel(fig, label, **kwargs):
    """Set xlabel for a plotly figure"""
    return fig.update_xaxes(title=label, **kwargs)


def ylabel(fig, label, **kwargs):
    """Set ylabel for a plotly figure"""
    return fig.update_yaxes(title=label, **kwargs)
