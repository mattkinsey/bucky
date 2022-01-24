"""Provide a nested version of dict() along with convenient API additions (apply, update, etc)."""
from collections.abc import Iterable, Mapping, MutableMapping  # pylint: disable=no-name-in-module
from copy import deepcopy

import yaml  # TODO replace pyyaml with ruamel.yaml everywhere?

# from pprint import pformat


def _is_dict_type(x):
    """Check is a variable has a dict-like interface."""
    return isinstance(x, MutableMapping)


class NestedDict(MutableMapping):
    """A nested version of dict."""

    def __init__(self, dict_of_dicts=None, seperator="."):
        """Init an empty dict or convert a dict of dicts into a NestedDict."""
        # TODO detect flattened dicts too?
        self.sep = seperator
        if dict_of_dicts is not None:
            if not _is_dict_type(dict_of_dicts):
                raise TypeError
            self._data = self.from_dict(dict_of_dicts)
        else:
            self._data = {}

    def __setitem__(self, key, value):
        """Setitem, doing it recusively if a flattened key is used."""
        if not isinstance(key, str):
            raise NotImplementedError(f"{self.__class__} only supports str-typed keys; for now")

        if self.sep in key:
            keys = key.split(self.sep)
            last_key = keys.pop()
            try:
                tmp = self._data
                for k in keys:
                    if k in tmp:
                        tmp = tmp[k]
                    else:
                        tmp[k] = self.__class__()
                        tmp = tmp[k]
                tmp[last_key] = value
            except KeyError as err:
                raise KeyError(key) from err
        else:
            self._data[key] = value

    def __getitem__(self, key):
        """Getitem, supports flattened keys."""
        if not isinstance(key, str):
            raise NotImplementedError(f"{self.__class__} only supports str-typed keys; for now")

        if self.sep in key:
            keys = key.split(self.sep)
            try:
                tmp = self._data
                for k in keys:
                    if k in tmp:
                        tmp = tmp[k]
            except KeyError as err:
                raise KeyError(key) from err
            return tmp
        else:
            return self._data[key]

    def __delitem__(self, key):
        """WIP."""
        # TODO this doesnt work for flattened keys
        del self._data[key]

    def __iter__(self):
        """WIP."""
        # provide a flatiter too?
        return iter(self._data)

    def __len__(self):
        """WIP."""
        # TODO
        return len(self._data)

    def __repr__(self):
        """REPL string representation for NestedDict, basically just yaml-ize it."""
        ret = "<" + self.__class__.__name__ + ">\n"
        # Just lean on yaml for now but it makes arrays very ugly
        ret += self.to_yaml(default_flow_style=None)
        # ret += pformat(self.to_dict())
        return ret

    # def __str__(self):

    def flatten(self, parent=""):
        """Flatten to a normal dict where the heirarcy exists in the key names."""
        ret = []
        for k, v in self.items():
            base_key = parent + self.sep + k if parent else k
            if _is_dict_type(v):  # isinstance(v, type(self)):
                ret.extend(v.flatten(parent=base_key).items())
            else:
                ret.append((base_key, v))

        return dict(ret)

    def from_flat_dict(self, flat_dict):
        """Create a NestedDict from a flattened dict."""
        ret = self.__class__()
        for k, v in flat_dict.items():
            ret[k] = v
        return ret

    def from_dict(self, dict_of_dicts):
        """Create a NestedDict from a dict of dicts."""
        ret = self.__class__()  # NestedDict()
        for k, v in dict_of_dicts.items():
            if _is_dict_type(v):
                ret[k] = self.from_dict(v)
            else:
                ret[k] = v
        return ret

    # def from_yaml(f):

    def to_dict(self):
        """Return self but as a proper dict of dicts."""
        ret = {}
        for k, v in self.items():
            if _is_dict_type(v):  # isinstance(v, type(self)):
                ret[k] = v.to_dict()
            else:
                ret[k] = v
        return ret

    def to_yaml(self, *args, **kwargs):
        """Return YAML represenation of self."""
        return yaml.dump(self.to_dict(), *args, **kwargs)

    def update(self, other=(), **kwargs):  # pylint: disable=arguments-differ
        """Update (like dict().update), but accept dict_of_dicts as input."""
        if other and isinstance(other, Mapping):
            for k, v in other.items():
                if _is_dict_type(v):
                    self[k] = self.get(k, self.__class__()).update(v)
                else:
                    self[k] = v

        elif other and isinstance(other, Iterable):
            for (k, v) in other:
                if _is_dict_type(v):
                    self[k] = self.get(k, self.__class__()).update(v)
                else:
                    self[k] = v

        for k, v in kwargs.items():
            if _is_dict_type(v):
                self[k] = self.get(k, self.__class__()).update(v)
            else:
                self[k] = v

        return self

    def apply(self, func, copy=False, key_filter=None, contains_filter=None):
        """Apply a function of values stored in self, optionally filtering or doing a deep copy."""
        ret = deepcopy(self) if copy else self

        for k, v in ret.items():
            if _is_dict_type(v):  # isinstance(v, type(ret)):
                if contains_filter is not None:
                    key_set = set((contains_filter,) if isinstance(contains_filter, str) else contains_filter)
                    if key_set.issubset(v.keys()):
                        ret[k] = func(v)
                        continue
                ret[k] = v.apply(func, copy=copy, key_filter=key_filter, contains_filter=contains_filter)
            else:
                if key_filter is not None:
                    if k == key_filter:
                        ret[k] = func(v)
                elif contains_filter is not None:
                    ret[k] = v
                else:
                    ret[k] = func(v)

        return ret


if __name__ == "__main__":

    test_dict = {"a": "a", "b": {"c": "c", "d": "d"}}
    print(test_dict)  # noqa: T001

    nd = NestedDict(test_dict)
    print(nd)  # noqa: T001

    up = {"b": {"d": 12}}
    print(nd.update(up))  # noqa: T001

    nd.apply(lambda x: x + "a" if isinstance(x, str) else x)
    print(nd)  # noqa: T001
