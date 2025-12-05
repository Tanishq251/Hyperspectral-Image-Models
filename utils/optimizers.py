"""Optimizer factory for flexible optimizer selection"""

import torch.optim as optim


# Default parameters for each optimizer
DEFAULT_PARAMS = {
    'adam': {
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.0  # Adam: no weight decay by default
    },
    'adamw': {
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.0001  # AdamW: weight decay enabled
    },
    'sgd': {
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 0.0
    },
    'rmsprop': {
        'alpha': 0.99,
        'eps': 1e-8,
        'momentum': 0.0,
        'weight_decay': 0.0
    },
    'adagrad': {
        'eps': 1e-10,
        'weight_decay': 0.0
    },
    'adadelta': {
        'rho': 0.9,
        'eps': 1e-8,
        'weight_decay': 0.0
    }
}


def create_optimizer(model, optimizer_name='adam', learning_rate=0.001, **kwargs):
    """
    Create optimizer based on name and parameters
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'sgd', 'adamw', 'rmsprop', 'adagrad', 'adadelta')
        learning_rate: Learning rate
        **kwargs: Additional optimizer parameters (optional, will use defaults if not provided)
    
    Returns:
        optimizer: PyTorch optimizer instance
    """
    
    optimizer_name = optimizer_name.lower()
    
    # Get default parameters for this optimizer
    if optimizer_name not in DEFAULT_PARAMS:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. "
                        f"Available: {', '.join(DEFAULT_PARAMS.keys())}")
    
    # Start with defaults
    params = DEFAULT_PARAMS[optimizer_name].copy()
    
    # Override with user-provided parameters
    params.update(kwargs)
    
    if optimizer_name == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=params['betas'],
            eps=params['eps'],
            weight_decay=params['weight_decay']
        )
    
    elif optimizer_name == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=params['betas'],
            eps=params['eps'],
            weight_decay=params['weight_decay']
        )
    
    elif optimizer_name == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=params['momentum'],
            weight_decay=params['weight_decay'],
            nesterov=params['nesterov']
        )
    
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            alpha=params['alpha'],
            eps=params['eps'],
            weight_decay=params['weight_decay'],
            momentum=params['momentum']
        )
    
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(
            model.parameters(),
            lr=learning_rate,
            eps=params['eps'],
            weight_decay=params['weight_decay']
        )
    
    elif optimizer_name == 'adadelta':
        return optim.Adadelta(
            model.parameters(),
            lr=learning_rate,
            rho=params['rho'],
            eps=params['eps'],
            weight_decay=params['weight_decay']
        )


def get_optimizer_info(optimizer_name):
    """Get information about an optimizer"""
    
    info = {
        'adam': {
            'description': 'Adaptive Moment Estimation - Good for most tasks',
            'params': ['betas', 'eps', 'weight_decay'],
            'default_lr': 0.001
        },
        'adamw': {
            'description': 'Adam with decoupled weight decay - Better generalization',
            'params': ['betas', 'eps', 'weight_decay'],
            'default_lr': 0.001
        },
        'sgd': {
            'description': 'Stochastic Gradient Descent - Classic, often best for CNNs',
            'params': ['momentum', 'nesterov', 'weight_decay'],
            'default_lr': 0.01
        },
        'rmsprop': {
            'description': 'RMSprop - Good for RNNs and non-stationary problems',
            'params': ['alpha', 'momentum', 'eps', 'weight_decay'],
            'default_lr': 0.001
        },
        'adagrad': {
            'description': 'Adaptive Gradient - Good for sparse data',
            'params': ['eps', 'weight_decay'],
            'default_lr': 0.01
        },
        'adadelta': {
            'description': 'Adadelta - No learning rate needed (uses adaptive lr)',
            'params': ['rho', 'eps', 'weight_decay'],
            'default_lr': 1.0
        }
    }
    
    return info.get(optimizer_name.lower(), {})


def list_optimizers():
    """List all available optimizers"""
    return ['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'adadelta']
