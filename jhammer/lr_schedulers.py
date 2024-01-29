import math

from jhammer.lr_utils import update_lr


def poly_lr(optimizer, initial_lr, epoch, max_epochs, min_lr=0, exponent=0.9):
    lr = initial_lr * (1 - epoch / max_epochs) ** exponent
    lr = lr if lr > min_lr else min_lr
    update_lr(lr, optimizer)
    return lr


def half_cycle_cosine_after_warmup_lr(optimizer, initial_lr, epoch, max_epochs, warmup_epochs, min_lr=0):
    """
    Decay the learning rate with half-cycle cosine after warmup
    Args:
        optimizer:
        initial_lr: the initial lr
        epoch: current epoch
        max_epochs: max epoch
        warmup_epochs: warmup after assigned epochs
        min_lr:

    Returns:

    """
    if epoch < warmup_epochs:
        lr = initial_lr * epoch / warmup_epochs
    else:
        lr = min_lr + (initial_lr - min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    update_lr(lr, optimizer)
    return lr
