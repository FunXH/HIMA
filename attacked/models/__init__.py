from .base import *
from .common import *

_MODEL_EXPORTS = {
    "DCMHT": (".DCMHT", "DCMHT"),
    "Baseline": (".baseline", "Baseline"),
    "DSPH": (".DSPH", "DSPH"),
    "DNPH": (".DNPH", "DNPH"),
}


def __getattr__(name):
    if name not in _MODEL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module_name, attr_name = _MODEL_EXPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
