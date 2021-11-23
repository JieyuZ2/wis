import hashlib
import numbers
import random
from typing import Union, List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .mytype import Config


def update_config(config: Config, args):
    config_dict = config._asdict()
    for k, v in config_dict.items():
        if hasattr(args, k):
            new_v = getattr(args, k)
            if new_v is not None:
                config_dict[k] = new_v
    return config_dict


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def normalized_entropy(p):
    return -((p * p.log()).mean(dim=-1))


def softmax(inputs, tau=1):
    logits = inputs / tau
    logits = torch.softmax(logits, dim=-1)
    return logits


def replace_array(a: np.ndarray, d: Union[Dict, List[Tuple]]) -> np.ndarray:
    t = np.inf * np.ones_like(a)
    if isinstance(d, dict):
        d = list(d.items())
    for m, n in d:
        t[a == m] = n
    inf_idx = t == np.inf
    t[inf_idx] = a[inf_idx]
    return t.astype(a.dtype)


def _hash(i: int) -> int:
    """Deterministic hash function."""
    byte_string = str(i).encode("utf-8")
    return int(hashlib.sha1(byte_string).hexdigest(), 16)


def probs_to_preds(
        probs: np.ndarray, tie_break_policy: str = "random", tol: float = 1e-5
) -> np.ndarray:
    """Convert an array of probabilistic labels into an array of predictions.

    Policies to break ties include:
    "abstain": return an abstain vote (-1)
    "true-random": randomly choose among the tied options
    "random": randomly choose among tied option using deterministic hash
    "first": always output the first class

    NOTE: if tie_break_policy="true-random", repeated runs may have slightly different results due to difference in broken ties

    Parameters
    ----------
    prob
        A [num_datapoints, num_classes] array of probabilistic labels such that each
        row sums to 1.
    tie_break_policy
        Policy to break ties when converting probabilistic labels to predictions
    tol
        The minimum difference among probabilities to be considered a tie

    Returns
    -------
    np.ndarray
        A [n] array of predictions (integers in [0, ..., num_classes - 1])

    Examples
    --------
    >>> probs_to_preds(np.array([[0.5, 0.5, 0.5]]), tie_break_policy="abstain")
    array([-1])
    >>> probs_to_preds(np.array([[0.8, 0.1, 0.1]]))
    array([0])
    """
    num_datapoints, num_classes = probs.shape
    if num_classes <= 1:
        raise ValueError(
            f"probs must have probabilities for at least 2 classes. "
            f"Instead, got {num_classes} classes."
        )

    Y_pred = np.empty(num_datapoints)
    diffs = np.abs(probs - probs.max(axis=1).reshape(-1, 1))

    for i in range(num_datapoints):
        max_idxs = np.where(diffs[i, :] < tol)[0]
        # max_idxs = np.where(diffs[i, :] == 0)[0]
        if len(max_idxs) == 1:
            Y_pred[i] = max_idxs[0]
        # Deal with "tie votes" according to the specified policy
        elif tie_break_policy == "random":
            Y_pred[i] = max_idxs[_hash(i) % len(max_idxs)]
        elif tie_break_policy == "true-random":
            Y_pred[i] = np.random.choice(max_idxs)
        elif tie_break_policy == "abstain":
            Y_pred[i] = -1
        elif tie_break_policy == "first":
            Y_pred[i] = max_idxs[0]
        else:
            raise ValueError(
                f"tie_break_policy={tie_break_policy} policy not recognized."
            )
    return Y_pred.astype(np.int)


def change_params_prefix(prefix: str, params: Union[Config, Dict]) -> Dict:
    if isinstance(params, Config):
        params = params._asdict()
    return {prefix + key: value for key, value in params.items()}


def cartesian(cardinalities, out=None):
    """Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    arrays = [np.asarray(range(card)) for card in cardinalities]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


def cross_entropy_with_probs(
        input: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
) -> torch.Tensor:
    """Calculate cross-entropy loss when targets are probabilities (floats), not ints.

    PyTorch's F.cross_entropy() method requires integer labels; it does accept
    probabilistic labels. We can, however, simulate such functionality with a for loop,
    calculating the loss contributed by each class and accumulating the results.
    Libraries such as keras do not require this workaround, as methods like
    "categorical_crossentropy" accept float labels natively.

    Note that the method signature is intentionally very similar to F.cross_entropy()
    so that it can be used as a drop-in replacement when target labels are changed from
    from a 1D tensor of ints to a 2D tensor of probabilities.

    Parameters
    ----------
    input
        A [num_points, num_classes] tensor of logits
    target
        A [num_points, num_classes] tensor of probabilistic target labels
    weight
        An optional [num_classes] array of weights to multiply the loss by per class
    reduction
        One of "none", "mean", "sum", indicating whether to return one loss per data
        point, the mean loss, or the sum of losses

    Returns
    -------
    torch.Tensor
        The calculated loss

    Raises
    ------
    ValueError
        If an invalid reduction keyword is submitted
    """
    if input.shape[1] == 1:
        input = input.squeeze()
        if target.ndim == 2:
            target = target[:, 1]
        return F.binary_cross_entropy_with_logits(input, target, weight=weight, reduction=reduction)
    else:
        target = target.squeeze()
        if target.ndim == 1:
            return F.cross_entropy(input, target.long(), weight=weight, reduction=reduction)

        num_points, num_classes = input.shape
        # Note that t.new_zeros, t.new_full put tensor on same device as t
        cum_losses = input.new_zeros(num_points)
        for y in range(num_classes):
            target_temp = input.new_full((num_points,), y, dtype=torch.long)
            y_loss = F.cross_entropy(input, target_temp, reduction="none")
            if weight is not None:
                y_loss = y_loss * weight[y]
            cum_losses += target[:, y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")
