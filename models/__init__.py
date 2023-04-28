
__all__ = [
    'loss',
    'disabled_train',
]


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


