"""HMM regime detection (Phase 2)."""

import numpy as np
from hmmlearn import hmm


def train_hmm(features, n_components: int):
    """Train GaussianHMM on feature matrix, return fitted model."""
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=1000,
        random_state=42,
    )
    model.fit(features)
    return model


def select_n_regimes(features, k_range=range(2, 6)):
    """Select optimal K using BIC over k_range, return best K and BIC scores dict."""
    bic_scores = {}
    n = len(features)
    for k in k_range:
        model = train_hmm(features, k)
        # BIC = -2 * log-likelihood + k_params * log(n)
        # Number of free params for full covariance GaussianHMM:
        # transition: k*(k-1), means: k*d, covariances: k*d*(d+1)/2
        d = features.shape[1]
        k_params = k * (k - 1) + k * d + k * d * (d + 1) // 2
        bic_scores[k] = -2 * model.score(features) * n + k_params * np.log(n)
    best_k = min(bic_scores, key=bic_scores.get)
    return best_k, bic_scores


def get_viterbi_path(model, features):
    """Return Viterbi regime label sequence for the feature matrix."""
    return model.predict(features)
