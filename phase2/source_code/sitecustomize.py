"""Universal Python 3.10+ / 3.13 compatibility shim for DGL and scientific libraries."""
import collections
import collections.abc

for _name in ("Mapping", "MutableMapping", "Iterable", "Iterator", "Sequence",
              "MutableSequence", "Callable", "Container", "Set", "MutableSet"):
    if not hasattr(collections, _name) and hasattr(collections.abc, _name):
        setattr(collections, _name, getattr(collections.abc, _name))
