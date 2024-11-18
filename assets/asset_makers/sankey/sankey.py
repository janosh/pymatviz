# %%
import pandas as pd
from mp_api.client import MPRester
from mp_api.client.core import MPRestError

import pymatviz as pmv
from pymatviz.enums import Key


pmv.set_plotly_template("pymatviz_white")


# %% Sankey diagram of crystal systems and space groups
try:
    with MPRester(use_document_model=False) as mpr:
        fields = [Key.mat_id, "symmetry.crystal_system", "symmetry.symbol"]
    docs = mpr.materials.summary.search(
        num_elements=(1, 3), fields=fields, num_chunks=30, chunk_size=1000
    )
except MPRestError:
    raise SystemExit(0) from None


# %%
df_mp = pd.json_normalize(docs).set_index(Key.mat_id)
df_mp.columns = [Key.crystal_system, Key.spg_symbol]

frequent_symbols = df_mp[Key.spg_symbol].value_counts().nlargest(20).index

df_spg = df_mp.query(f"{Key.spg_symbol} in @frequent_symbols")


# %%
fig = pmv.sankey_from_2_df_cols(
    df_spg, [Key.crystal_system, Key.spg_symbol], labels_with_counts="percent"
)
title = "Common Space Groups in Materials Project"
fig.layout.title = dict(text=title, x=0.5, y=0.95)
fig.layout.margin.t = 50
fig.show()
pmv.io.save_and_compress_svg(fig, "sankey-crystal-sys-to-spg-symbol")
