import math

from jhammer.lr_utils import update_lr


def poly_lr(optimizer, initial_lr, epoch, max_epochs, min_lr=0, exponent=0.9):
    lr = initial_lr * (1 - epoch / max_epochs) ** exponent
    lr = lr if lr > min_lr else min_lr
    update_lr(lr, optimizer)
    return lr


def linear_warmup_cosine_decay_lr(optimizer, initial_lr, epoch, max_epochs, warmup_epochs, min_lr=0):
    """
    linear warmup and then cosine decay.
    Args:
        optimizer (torch.optim.Optimizer):
        initial_lr (float): The initial lr.
        epoch (int): Current epoch.
        max_epochs (int): Max epoch.
        warmup_epochs (int): Warmup after assigned epochs.
        min_lr (float, optional, default=0): The minimal lr.

    Returns:

    """

    if epoch < warmup_epochs:
        lr = initial_lr * epoch / warmup_epochs
    else:
        lr = min_lr + (initial_lr - min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    update_lr(lr, optimizer)
    return lr
