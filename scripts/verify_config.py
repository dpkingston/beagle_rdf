#!/usr/bin/env python3
# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Verify node configuration files before deployment.

Usage
-----
  scripts/verify_config.py config/dpk-tdoa1.yaml config/kb7ryy.yaml
  scripts/verify_config.py path/to/config.json

Per file:
  * Detects YAML vs JSON by extension (.yaml/.yml vs .json) and content.
  * Validates against ``NodeConfig`` (the same pydantic schema the node
    uses at runtime).  Catches type errors, missing required fields,
    out-of-range values, etc.
  * **In addition**, scans recursively for any keys NOT declared on the
    matching pydantic model.  This is the check that the runtime
    deliberately doesn't enforce (to avoid bricking the fleet on a
    rolling config update), but that catches typos like
    ``carrier_onset_margin_db`` written inside the ``carrier`` section
    where the field is actually just ``onset_margin_db``.

Exit code is the number of files that failed verification.

This script is intended to be the first line of defense before any
config change reaches the running fleet — operators should pipe new
configs through it before applying via the admin API or pushing to
the server's registry DB.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make the in-repo node package importable when running from a checkout.
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None  # YAML is optional; JSON-only configs still work without it.

try:
    from pydantic import BaseModel, ValidationError
except ImportError as exc:
    print(f"ERROR: pydantic not available: {exc}", file=sys.stderr)
    print("Run from the project's virtualenv: env/bin/python scripts/verify_config.py ...",
          file=sys.stderr)
    sys.exit(2)

from beagle_node.config.schema import NodeConfig  # noqa: E402


# ---------------------------------------------------------------------------
# File loading (YAML or JSON)
# ---------------------------------------------------------------------------

def _load_file(path: Path) -> Any:
    """Load a config file as a Python dict.

    YAML is a superset of JSON for our purposes; ``yaml.safe_load`` handles
    both.  We fall back to ``json.loads`` if PyYAML isn't installed.
    """
    text = path.read_text()
    if yaml is not None:
        return yaml.safe_load(text)
    # Without yaml, accept JSON only.
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"{path}: PyYAML not installed; cannot parse non-JSON files. "
            f"(JSON parse also failed: {exc})"
        )


# ---------------------------------------------------------------------------
# Unknown-key detection (the strict check the runtime skips)
# ---------------------------------------------------------------------------

def _model_field_names(model: type[BaseModel]) -> set[str]:
    """Return the set of declared field names on a pydantic model."""
    return set(model.model_fields.keys())


def _annotation_model(ann: Any) -> type[BaseModel] | None:
    """If ``ann`` (a pydantic field annotation) is a BaseModel subclass or
    wraps one (Optional, list of, etc.), return that model class.
    Otherwise return None — we don't recurse into non-model values.
    """
    # Strip Optional / Union / list / etc.
    origin = getattr(ann, "__origin__", None)
    if origin is None:
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            return ann
        return None
    # Composite annotations — walk their args looking for a BaseModel.
    args = getattr(ann, "__args__", ()) or ()
    for arg in args:
        nested = _annotation_model(arg)
        if nested is not None:
            return nested
    return None


def _scan_unknown_keys(
    data: Any,
    model: type[BaseModel],
    path: str = "",
    errors: list[str] | None = None,
) -> list[str]:
    """Recursively walk ``data`` (a parsed dict) checking each level's keys
    against the corresponding pydantic model's declared fields.

    Reports every unknown key found, with a misspelling suggestion when a
    field name is close to what the operator typed.
    """
    if errors is None:
        errors = []
    if not isinstance(data, dict):
        return errors
    declared = _model_field_names(model)
    for key, value in data.items():
        if key not in declared:
            suggestion = difflib.get_close_matches(key, sorted(declared), n=1, cutoff=0.6)
            hint = f"  did you mean '{suggestion[0]}'?" if suggestion else ""
            errors.append(
                f"{path}{key}: unknown field on {model.__name__}.{hint}"
            )
            continue
        # Known field — descend if it's another pydantic model.
        ann = model.model_fields[key].annotation
        nested_model = _annotation_model(ann)
        if nested_model is not None:
            _scan_unknown_keys(
                value, nested_model, path=f"{path}{key}.", errors=errors,
            )
    return errors


# ---------------------------------------------------------------------------
# Per-file verification
# ---------------------------------------------------------------------------

def verify_one(path: Path, *, verbose: bool = False) -> bool:
    """Verify a single config file.  Returns True on success, False on failure.
    Failure details are printed to stderr."""
    print(f"=== {path} ===")

    # 1) Load.
    if not path.exists():
        print(f"  FAIL: file not found", file=sys.stderr)
        return False
    try:
        data = _load_file(path)
    except Exception as exc:
        print(f"  FAIL: parse error: {exc}", file=sys.stderr)
        return False
    if not isinstance(data, dict):
        print(f"  FAIL: top-level must be a mapping, got {type(data).__name__}",
              file=sys.stderr)
        return False

    # 2) Schema validation (the same pydantic runs at startup).
    schema_errors: list[str] = []
    try:
        config = NodeConfig.model_validate(data)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            schema_errors.append(f"  {loc}: {err['msg']} [{err['type']}]")
        config = None

    # 3) Unknown-key scan (the check the runtime deliberately omits).
    extra_errors = _scan_unknown_keys(data, NodeConfig)

    # 4) Report.
    if schema_errors:
        print(f"  schema validation errors ({len(schema_errors)}):",
              file=sys.stderr)
        for e in schema_errors:
            print(e, file=sys.stderr)
    if extra_errors:
        print(f"  unknown-field errors ({len(extra_errors)}):",
              file=sys.stderr)
        for e in extra_errors:
            print(f"  {e}", file=sys.stderr)

    if not schema_errors and not extra_errors:
        print(f"  OK")
        if verbose and config is not None:
            print(f"  node_id:        {config.node_id}")
            print(f"  sync_mode:      {config.sync.mode if hasattr(config, 'sync') else '?'}")
            print(f"  sdr_mode:       {config.sdr_mode if hasattr(config, 'sdr_mode') else '?'}")
            print(f"  target_count:   "
                  f"{len(config.target_channels) if hasattr(config, 'target_channels') else '?'}")
        return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify one or more node configuration files (YAML or JSON) for "
            "syntax errors, schema violations, and unknown / misspelled "
            "field names."
        ),
        epilog=(
            "Exit code is the number of files that failed verification "
            "(0 means all good; can be fed straight into shell pipelines)."
        ),
    )
    parser.add_argument("files", nargs="+", type=Path,
                        help="Config files to verify (any mix of .yaml/.yml/.json).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="On success, print a brief summary of each loaded config.")
    args = parser.parse_args(argv)

    n_failed = 0
    for f in args.files:
        if not verify_one(f, verbose=args.verbose):
            n_failed += 1
        print()

    if n_failed == 0:
        print(f"All {len(args.files)} config file(s) passed verification.")
    else:
        print(f"FAILED: {n_failed} of {len(args.files)} file(s) had errors.",
              file=sys.stderr)
    return n_failed


if __name__ == "__main__":
    sys.exit(main())
