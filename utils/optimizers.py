"""Optimizer factory for flexible optimizer selection.

Supported: adam, adamw, sgd, rmsprop, adagrad, adadelta

Usage from config:
    training:
      optimizer: "adamw"
      optimizer_params:
        weight_decay: 0.01
"""

import torch.optim as optim


# ──────────────────────────────────────────────
# Registry: optimizer class + sensible defaults
# ──────────────────────────────────────────────
_REGISTRY = {
    'adam': {
        'class': optim.Adam,
        'defaults': {
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.0,
        },
        'description': 'Adaptive Moment Estimation — good for most tasks',
        'default_lr': 0.001,
    },
    'adamw': {
        'class': optim.AdamW,
        'defaults': {
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 1e-4,
        },
        'description': 'Adam with decoupled weight decay — better generalization',
        'default_lr': 0.001,
    },
    'sgd': {
        'class': optim.SGD,
        'defaults': {
            'momentum': 0.9,
            'nesterov': True,
            'weight_decay': 0.0,
        },
        'description': 'Stochastic Gradient Descent — classic, often best for CNNs',
        'default_lr': 0.01,
    },
    'rmsprop': {
        'class': optim.RMSprop,
        'defaults': {
            'alpha': 0.99,
            'eps': 1e-8,
            'momentum': 0.0,
            'weight_decay': 0.0,
        },
        'description': 'RMSprop — good for RNNs and non-stationary problems',
        'default_lr': 0.001,
    },
    'adagrad': {
        'class': optim.Adagrad,
        'defaults': {
            'eps': 1e-10,
            'weight_decay': 0.0,
        },
        'description': 'Adaptive Gradient — good for sparse data',
        'default_lr': 0.01,
    },
    'adadelta': {
        'class': optim.Adadelta,
        'defaults': {
            'rho': 0.9,
            'eps': 1e-8,
            'weight_decay': 0.0,
        },
        'description': 'Adadelta — adaptive learning rate, lr param barely matters',
        'default_lr': 1.0,
    },
}


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def create_optimizer(model, optimizer_name='adam', learning_rate=0.001, **kwargs):
    """Create a PyTorch optimizer by name.

    Args:
        model: PyTorch model (or any object with ``.parameters()``).
        optimizer_name: One of 'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'adadelta'.
        learning_rate: Learning rate.
        **kwargs: Override any default parameter for the chosen optimizer.

    Returns:
        torch.optim.Optimizer instance.
    """
    name = optimizer_name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown optimizer: '{optimizer_name}'. "
            f"Available: {', '.join(_REGISTRY)}"
        )

    entry = _REGISTRY[name]
    params = {**entry['defaults'], **kwargs}  # defaults + user overrides

    return entry['class'](model.parameters(), lr=learning_rate, **params)


def get_optimizer_info(optimizer_name):
    """Return metadata dict for an optimizer (description, default params, default lr)."""
    entry = _REGISTRY.get(optimizer_name.lower())
    if entry is None:
        return {}
    return {
        'description': entry['description'],
        'params': list(entry['defaults'].keys()),
        'default_lr': entry['default_lr'],
        'defaults': entry['defaults'],
    }


def list_optimizers():
    """Return list of available optimizer names."""
    return list(_REGISTRY.keys())
