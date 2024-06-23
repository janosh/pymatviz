# %%
import numpy as np
import pandas as pd

from pymatviz.io import save_and_compress_svg
from pymatviz.sankey import sankey_from_2_df_cols


# %% Sankey diagram of random integers
cols = ["col_a", "col_b"]
np_rng = np.random.default_rng(seed=0)
df_rand_ints = pd.DataFrame(np_rng.integers(1, 6, size=(100, 2)), columns=cols)
fig = sankey_from_2_df_cols(df_rand_ints, cols, labels_with_counts="percent")
rand_int_title = "Two sets of 100 random integers from 1 to 5"
fig.update_layout(title=dict(text=rand_int_title, x=0.5, y=0.87))
code_anno = dict(
    x=0.5,
    y=-0.2,
    text="<span style='font-family: monospace;'>df = pd.DataFrame("
    "np_rng.integers(1, 6, size=(100, 2)), columns=['col_a','col_b'])<br>"
    "fig = sankey_from_2_df_cols(df, df.columns)</span>",
    font_size=12,
    showarrow=False,
)
fig.add_annotation(code_anno)
save_and_compress_svg(fig, "sankey-from-2-df-cols-randints")
