#-*- coding: utf-8 -*-

import argparse
import numpy as np
import tomlkit


def parse_toml(tomlfile, default="default.toml"):

    # First load defaults
    with open(default, "r") as fid:
        cfg = tomlkit.load(fid)

    # Load specific configuration
    with open(tomlfile, "r") as fid:
        cfg_new = tomlkit.load(fid)

    # Update sections
    for key, value in cfg.items():
        sub_cfg = cfg[key]
        try:
            sub_new = cfg_new[key]
            sub_cfg.update(sub_new)
        except KeyError:
            pass
        except NonExistentKey:
            pass

    # Convert all values to POPO
    cfg_popo = {}
    for section_name, section in cfg.items():
        cfg_popo[section_name] = {}
        for key, value in section.items():
            cfg_popo[section_name][key] = tomlkit_to_popo(value)

    # Return struct format
    return DictToStruct(cfg_popo)


class DictToStruct:
    """
    Convenience class for converting dict to struct-like object.
    """
    def __init__(self, data_dict):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                self.__dict__[key] = DictToStruct(value)
            else:
                self.__dict__[key] = value

    def set(self, key, value):
        """
        Manually set key-value pair.
        """
        setattr(self, key, value)

    def __delattr__(self, *args, **kwargs):
        raise AttributeError("DictToStruct attributes cannot be deleted.")

    def __setattr__(self, *args, **kwargs):
        raise AttributeError(
            "DictToStruct attributes cannot be assigned. Use set(key, value)."
        )

    def todict(self):
        out = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictToStruct):
                out[key] = value.todict()
            else:
                out[key] = value
        return out


def parse_args(description='Run program.'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('cfgfile', type=str, help='Configuration file.')
    parser.add_argument('--restore', action='store_true', help='Restore checkpoints.')
    return parser.parse_known_args()


def update_cfg_with_extra(cfg, extra_args):
    if len(extra_args) < 2:
        return cfg
    cdict = cfg.todict()
    n_args = len(extra_args) // 2
    for i in range(n_args):
        key_str = extra_args[2*i][2:]
        value_str = extra_args[2*i + 1]
        keys = key_str.split('.')
        try:
            original_value = cdict[keys[0]][keys[1]]
            cdict[keys[0]][keys[1]] = type(original_value)(value_str)
        except KeyError:
            print(f'Could not find cfg attribute {key_str}. Skipping.')
            continue
    return DictToStruct(cdict)


def tomlkit_to_popo(d):
    """
    Hack from https://github.com/sdispater/tomlkit/issues/43
    """
    try:
        result = getattr(d, "value")
    except AttributeError:
        result = d

    if isinstance(result, list):
        result = [tomlkit_to_popo(x) for x in result]
    elif isinstance(result, dict):
        result = {
            tomlkit_to_popo(key): tomlkit_to_popo(val) for key, val in result.items()
        }
    elif isinstance(result, tomlkit.items.Integer):
        result = int(result)
    elif isinstance(result, tomlkit.items.Float):
        result = float(result)
    elif isinstance(result, tomlkit.items.String):
        result = str(result)
    elif isinstance(result, tomlkit.items.Bool):
        result = bool(result)

    return result


# end of file
