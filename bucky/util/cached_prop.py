"""Cached property descript that will only eval on first access.

From https://stackoverflow.com/questions/4037481/caching-class-attributes-in-python
"""
import sys

ALLOW_FUNCTOOLS = sys.version_info >= (3, 8, 0)

if ALLOW_FUNCTOOLS:
    from functools import cached_property  # pylint: disable=unused-import
else:
    # TODO I think this was added to functools in 3.8 or 3.9... -Matt
    class cached_property:  # pylint: disable=too-few-public-methods
        """Descriptor (non-data) for building an attribute on-demand on first use."""

        def __init__(self, factory):
            """Init the function as a factory so that factory(instance) will build the attribute."""
            self._attr_name = factory.__name__
            self._factory = factory

        def __get__(self, instance, owner):
            """Get either the evaluated property or its cached value."""
            # Build the attribute.
            attr = self._factory(instance)

            # Cache the value; hide ourselves.
            setattr(instance, self._attr_name, attr)

            return attr
