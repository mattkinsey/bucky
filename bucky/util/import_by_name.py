import importlib


def import_by_name(fstr):
    mod_name, func_name = fstr.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func
