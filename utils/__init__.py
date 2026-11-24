from .utils import *
from .scheduler import PolyLR
from .loss import FocalLoss


def __getattr__(name):
    # Visdom is optional for preprocessing, evaluation and inference.
    if name == 'Visualizer':
        from .visualizer import Visualizer
        return Visualizer
    raise AttributeError(name)
