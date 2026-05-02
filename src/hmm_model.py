"""HMM regime detection (Phase 2). Stubs only — not implemented yet."""


def train_hmm(features, n_components: int):
    """Train GaussianHMM on feature matrix, return fitted model."""
    raise NotImplementedError("Phase 2")


def select_n_regimes(features, k_range=range(2, 6)):
    """Select optimal K using BIC over k_range, return best K and BIC scores."""
    raise NotImplementedError("Phase 2")


def get_viterbi_path(model, features):
    """Return Viterbi regime label sequence for the feature matrix."""
    raise NotImplementedError("Phase 2")
