# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
WarnOnUnknownFieldsBase — runtime warning for unknown pydantic fields.

This is the runtime counterpart to ``scripts/verify_config.py``.

Pydantic v2's default ``extra="ignore"`` silently drops unknown fields,
which is great for forward compatibility (older nodes can keep sending
events with retired fields; newer configs can carry deprecated keys
during a rolling cleanup) but it also means a config typo never
surfaces — the affected setting just keeps the default forever.

This base class restores the visibility without changing the safety
posture:

  * Unknown fields are still **dropped, not stored**, just like
    ``extra="ignore"``.  The node / server keeps running.
  * The first time any ``(model, field)`` pair is seen, we log a
    ``WARNING``.  Subsequent occurrences are suppressed so a high-rate
    event source (a legacy node firing 100 events / hour with a stale
    field) doesn't flood the log.
  * Operators are pointed at ``scripts/verify_config.py`` for batch
    cleanup before deployment.

The fleet-safety property the project requires is preserved: a typo in
a config pushed to a node will not prevent the node from starting or
processing data, only log a warning the operator can act on.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, model_validator

logger = logging.getLogger(__name__)


class WarnOnUnknownFieldsBase(BaseModel):
    """BaseModel subclass that logs a one-time warning per unknown field
    seen, then drops it.

    Inherit instead of ``BaseModel`` for every pydantic model that
    deserializes operator-authored configs or wire-format event payloads.
    """

    model_config = ConfigDict(extra="ignore")

    # Process-wide deduplication of warnings.  Each (class_name, field_name)
    # pair triggers at most one log line per process lifetime.  Resetting
    # is only useful in tests (see _reset_warned_keys below).
    _warned_keys: ClassVar[set[tuple[str, str]]] = set()

    @model_validator(mode="before")
    @classmethod
    def _warn_on_unknown_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            declared = set(cls.model_fields.keys())
            for key in data:
                if key in declared:
                    continue
                cache_key = (cls.__name__, key)
                if cache_key in WarnOnUnknownFieldsBase._warned_keys:
                    continue
                WarnOnUnknownFieldsBase._warned_keys.add(cache_key)
                logger.warning(
                    "Unknown field '%s' on %s -- accepting but ignoring "
                    "(legacy compatibility).  Run scripts/verify_config.py "
                    "to vet configs before deployment.",
                    key, cls.__name__,
                )
        return data


def _reset_warned_keys() -> None:
    """Clear the warned-keys cache.  For tests only — production code
    should never need this.
    """
    WarnOnUnknownFieldsBase._warned_keys.clear()
