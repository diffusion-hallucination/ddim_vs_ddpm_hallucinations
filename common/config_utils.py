from __future__ import annotations

from argparse import Namespace

from omegaconf import OmegaConf


def to_plain_config(cfg):
    # to plain config.
    # this supports the shared config helpers used by the paper's training and evaluation entrypoints.
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, Namespace):
        return vars(cfg)
    return cfg


def cfg_get(cfg, key: str, default=None):
    # cfg get.
    # this supports the shared config helpers used by the paper's training and evaluation entrypoints.
    current = cfg
    for part in str(key).split("."):
        if OmegaConf.is_config(current):
            if part not in current:
                return default
            current = current[part]
            continue
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
            continue
        if isinstance(current, Namespace):
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
            continue
        return default
    return current
