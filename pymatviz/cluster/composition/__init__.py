"""Chemical clustering module for material composition analysis.

This module provides utilities for clustering and visualizing materials based on their
chemical composition.
"""

from pymatviz.cluster.composition.embed import matminer_featurize, one_hot_encode
from pymatviz.cluster.composition.plot import (
    EmbeddingMethod,
    ProjectionMethod,
    cluster_compositions,
)
from pymatviz.cluster.composition.project import project_vectors
