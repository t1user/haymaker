"""Documentation-only adapters for public third-party type names."""

from typing import Any


class PublicTypeFormatter:
    """Provide a pickleable Sphinx formatter; bare function config is not cached."""

    def __call__(self, annotation: Any, config: Any = None) -> str | None:
        """Link public types, leaving all other annotation formatting to Sphinx."""
        names = {
            "pandas.core.frame.DataFrame": "pandas.DataFrame",
            "pandas.core.series.Series": "pandas.Series",
            "numpy.random._generator.Generator": "numpy.random.Generator",
            "pd.DataFrame": "pandas.DataFrame",
            "pd.Series": "pandas.Series",
        }
        name = (
            annotation
            if isinstance(annotation, str)
            else getattr(annotation, "__forward_arg__", None)
            or f"{getattr(annotation, '__module__', '')}.{getattr(annotation, '__qualname__', '')}"
        )
        if public_name := names.get(name):
            return f":py:class:`{public_name}`"
        # Eventkit indexes Event, but not its Timer implementation.
        if name == "eventkit.ops.create.Timer":
            return "``eventkit.Timer`` (:class:`eventkit.event.Event`)"
        return None
