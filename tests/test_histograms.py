from ml_matrics import residual_hist, spacegroup_hist, true_pred_hist

from . import y_pred, y_true


def test_residual_hist():
    residual_hist(y_true, y_pred)


def test_true_pred_hist():
    true_pred_hist(y_true, y_pred, y_true - y_pred)


def test_spacegroup_hist():
    spacegroup_hist(range(1, 231))
    spacegroup_hist(range(1, 231), show_counts=False)
    spacegroup_hist(range(1, 231), show_minor_xticks=True)
