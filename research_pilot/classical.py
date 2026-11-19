"""Classical forecast methods. Model inference receives only allowed observations and graph edges."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pilot import E, I, P, S, HORIZON, HISTORY, KNOWN_SCORE, TRAIN_ORIGINS

MASK_COLUMNS = (4, 5, 12, 13)
METHODS = ("prevalence", "degree", "observed_neighbor", "mask_only",
           "logistic", "boosted_trees", "fitted_mechanistic")


def adjacency(edges, nodes=256):
    u, v = edges.T
    a, b = np.r_[u, v], np.r_[v, u]
    return csr_matrix((np.ones(len(a)), (a, b)), shape=(nodes, nodes))


def design(x, origins, adj):
    """Current eight channels, window means, and graph aggregates of visible evidence."""
    parts = []
    for t in origins:
        w = x[t - HISTORY + 1:t + 1]
        if len(w) != HISTORY:
            raise ValueError("incomplete observation window")
        current, mean = w[-1], w.mean(0)
        parts.append(np.column_stack([current, mean, adj @ current[:, [E, I, 4]],
                                      adj @ mean[:, [E, I, 4]]]))
    return np.asarray(parts, dtype=np.float32)


def columns(name):
    if name == "degree":
        return [7]
    if name == "observed_neighbor":
        return [17]  # count of currently visible infectious neighbors
    if name == "mask_only":
        return list(MASK_COLUMNS)
    return slice(None)


def classifier(name, setting):
    if name == "boosted_trees":
        return HistGradientBoostingClassifier(max_leaf_nodes=setting, max_iter=90,
                                              learning_rate=0.08, l2_regularization=1.0,
                                              early_stopping=False, random_state=1701)
    return make_pipeline(StandardScaler(), LogisticRegression(C=setting, max_iter=300))


@dataclass
class Dynamics:
    """Effective hazards fitted from fully visible training histories."""
    hazard: float
    progress: float
    remove: float
    prior: list

    @classmethod
    def fit(cls, training):
        susceptible, events, exposed, progressed, infectious, removed = [], [], 0, 0, 0, 0
        prior = np.zeros(4)
        for x, adj in training:
            if not np.all(x[:, :, 4] == 1):
                raise ValueError("fit process parameters on fully observed training views only")
            state = np.argmax(x[:, :, :4], axis=2)
            a, b = state[:-1], state[1:]
            count_i = (adj @ (a == I).T).T
            susceptible.append(count_i[a == S])
            events.append((b != S)[a == S])
            exposed += np.sum(a == E)
            progressed += np.sum((a == E) & ((b == I) | (b == P)))
            infectious += np.sum(a == I)
            removed += np.sum((a == I) & (b == P))
            starts = np.array(TRAIN_ORIGINS) - HISTORY + 1
            prior += np.bincount(state[starts].ravel(), minlength=4)
        counts = np.concatenate(susceptible)
        y = np.concatenate(events)
        total = np.bincount(counts.astype(int))
        pos = np.bincount(counts[y].astype(int), minlength=len(total))
        if pos[0]:
            raise ValueError("unexplained first event at zero exposure in full training view")
        dose = np.arange(1, len(total))
        def loss(rate):
            return (np.dot(total[1:] - pos[1:], rate * dose)
                    - np.dot(pos[1:], np.log(-np.expm1(-rate * dose))))
        fitted = minimize_scalar(loss, bounds=(1e-8, 5), method="bounded")
        if not fitted.success:
            raise RuntimeError(fitted.message)
        return cls(float(fitted.x), float(progressed / exposed), float(removed / infectious),
                   (prior / prior.sum()).tolist())

    def advance(self, belief, adj):
        s, e, i, p = belief.T
        new = s * -np.expm1(-self.hazard * (adj @ i))
        progress = self.progress * (e + new)
        removed = self.remove * (i + progress)
        return np.column_stack((s - new, e + new - progress,
                                i + progress - removed, p + removed))

    def score(self, x, t, adj):
        w = x[t - HISTORY + 1:t + 1]
        belief = np.tile(self.prior, (w.shape[1], 1))
        for step, obs in enumerate(w):
            if step:
                belief = self.advance(belief, adj)
            visible = obs[:, 4].astype(bool)
            belief[visible] = obs[visible, :4]
            # Confirmation before this window rules out susceptibility.
            hidden_known = (~visible) & (obs[:, 6] == 1)
            belief[hidden_known, S] = 0
            belief[hidden_known] /= belief[hidden_known].sum(axis=1, keepdims=True)
        start = belief[:, S].copy()
        for _ in range(HORIZON):
            belief = self.advance(belief, adj)
        return np.clip(start - belief[:, S], 0, 1)


def predict(name, model, x, origins, adj, matrix):
    if name == "prevalence":
        raw = np.full(matrix.shape[:2], model)
    elif name == "fitted_mechanistic":
        raw = np.array([model.score(x, t, adj) for t in origins])
    else:
        flat = matrix.reshape(-1, matrix.shape[-1])
        raw = model.predict_proba(flat[:, columns(name)])[:, 1].reshape(matrix.shape[:2])
    if not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1:
        raise ValueError("invalid probability")
    return np.where(x[list(origins), :, 6] == 1, KNOWN_SCORE, raw)
