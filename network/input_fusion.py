"""Adapt the existing input convolution; no extra encoder or fusion branch."""

import torch
from torch import nn


def input_conv(model):
    """Return the state-dict prefix and first backbone convolution."""
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            return 'backbone' + ('.' + name if name else ''), module
    raise ValueError('Backbone has no input convolution')


def expand_rgb_input(model, channels):
    name, old = input_conv(model)
    if channels == old.in_channels:
        return
    if old.in_channels != 3 or channels != 4 or old.groups != 1:
        raise ValueError('Only an ordinary RGB convolution can be expanded to RGB-D')
    new = nn.Conv2d(channels, old.out_channels, old.kernel_size, old.stride,
                    old.padding, old.dilation, old.groups, old.bias is not None,
                    old.padding_mode).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    parent_name, child_name = name.rsplit('.', 1)
    setattr(model.get_submodule(parent_name), child_name, new)
