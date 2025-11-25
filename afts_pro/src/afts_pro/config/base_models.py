from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict


def _to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class BaseConfigModel(BaseModel):
    """
    Base configuration model with shared validation rules.
    """

    model_config: ConfigDict = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        validate_assignment=True,
        extra="forbid",
    )


AliasGenerator = Callable[[str], str]
