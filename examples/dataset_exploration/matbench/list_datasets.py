# %%
from matminer.datasets import get_available_datasets


# %%
matminer_datasets = get_available_datasets()

matbench_datasets = [dset for dset in matminer_datasets if dset.startswith("matbench_")]


print(f"total datasets = {len(matbench_datasets)}\n{matbench_datasets=}")
# 13 datasets in Matbench v0.1 as of Mar 2021:
#   dielectric, expt_gap, expt_is_metal, glass,
#   jdft2d, log_gvrh, log_kvrh, mp_e_form, mp_gap,
#   mp_is_metal, perovskites, phonons, steels",
