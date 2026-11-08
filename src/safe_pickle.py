"""Restricted unpickler for third-party pandas/numpy pickles (e.g. the HUST dataset).

Pickle can execute arbitrary code.  Only a closed allow-list of numpy/pandas/builtin
symbols is resolved; anything else raises.  The HUST files were additionally pre-screened
with pickletools (only numpy / pandas / builtins.slice references were found).
"""
import io
import pickle

_ALLOWED_PREFIXES = ("numpy", "pandas")
_ALLOWED_EXACT = {("builtins", "slice"), ("builtins", "dict"), ("builtins", "list"),
                  ("builtins", "tuple"), ("builtins", "set"), ("collections", "OrderedDict"),
                  ("copyreg", "_reconstructor")}


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(_ALLOWED_PREFIXES) or (module, name) in _ALLOWED_EXACT:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global: {module}.{name}")


def safe_load(path):
    with open(path, "rb") as f:
        return _Restricted(io.BytesIO(f.read())).load()
