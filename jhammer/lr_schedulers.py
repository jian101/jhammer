def poly_lr(initial_lr, epoch, max_epochs, min_lr=0, exponent=0.9):
    lr = initial_lr * (1 - epoch / max_epochs) ** exponent
    return lr if lr > min_lr else min_lr
