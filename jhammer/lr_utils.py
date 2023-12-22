def update_lr(lr: float, optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]
