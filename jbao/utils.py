# -*- coding: utf-8 -*-

import argparse
import numpy as np
import tomlkit
import time


def update_dict(original, updates):
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and key in original
            and isinstance(original[key], dict)
        ):
            # Recursively update sub-dictionaries
            update_dict(original[key], value)
        else:
            # Update the value in the original dictionary
            original[key] = value
    return original


def parse_toml(tomlfile, default=None):

    # Load defaults and then update if default provided
    if default is not None:
        with open(default, "r") as fid:
            cfg = tomlkit.load(fid)

        # Load specific configuration
        with open(tomlfile, "r") as fid:
            cfg_new = tomlkit.load(fid)

        # Update default
        cfg = update_dict(cfg, cfg_new)

    else:
        with open(tomlfile, "r") as fid:
            cfg = tomlkit.load(fid)

    # Convert all values to POPO
    cfg_popo = {}
    for section_name, section in cfg.items():
        try:
            cfg_popo[section_name] = {}
            for key, value in section.items():
                cfg_popo[section_name][key] = tomlkit_to_popo(value)
        except AttributeError:
            cfg_popo[section_name] = tomlkit_to_popo(section)

    # Return struct format
    return DictToStruct(cfg_popo)


def save_toml(cfg, tomlfile):
    with open(tomlfile, "w") as fid:
        tomlkit.dump(cfg.todict(), fid)


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

    def update(self, path, value):
        """
        Returns a new DictToStruct instance with the updated value.
        'path' should be a dot-separated string, e.g., 'user.profile.name'
        """
        # Convert the current object back to a dict
        data = self.todict()

        # Navigate/Update the dictionary
        keys = path.split('.')
        ref = data
        for key in keys[:-1]:
            ref = ref.setdefault(key, {})

        ref[keys[-1]] = value

        # Return a new instance
        return DictToStruct(data)

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


def parse_args(description="Run program."):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("cfgfile", type=str, help="Configuration file.")
    parser.add_argument("--restore", action="store_true", help="Restore checkpoints.")
    return parser.parse_known_args()


def update_cfg_with_extra(cfg, extra_args):
    if len(extra_args) < 2:
        return cfg
    cdict = cfg.todict()
    n_args = len(extra_args) // 2
    for i in range(n_args):
        key_str = extra_args[2 * i][2:]
        value_str = extra_args[2 * i + 1]
        keys = key_str.split(".")
        # Detect any special type specifiers
        if value_str[-2] == "@":
            if value_str[-1] == "f":
                value_type = float
            elif value_str[-1] == "i":
                value_type = int
            else:
                value_type = float
            value_str = value_str[:-2]
        else:
            try:
                original_value = cdict[keys[0]][keys[1]]
                value_type = type(original_value)
            except KeyError:
                print(f"Could not find cfg attribute {key_str}. Skipping.")
                continue
        if value_type == bool:
            if value_str.lower() in ("true", "1", "yes", "on"):
                save_value = True
            elif value_str.lower() in ("false", "0", "no", "off"):
                save_value = False
            else:
                raise ValueError(f"Invalid boolean: {value}")
        else:
            save_value = value_type(value_str)
        cdict[keys[0]][keys[1]] = save_value
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


class Timer:
    """
    A simple timer class for use in context managers.
    """

    def __init__(self, desc="Timing context"):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t0 = time.time()
        return

    def __exit__(self, *exc_args):
        tf = time.time()
        print(" - elapsed time: %f s" % (tf - self.t0))


# end of file
