#!/usr/bin/env python3
"""Load rules/rules_spec.yaml and validate against rules/rules_schema.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    try:
        import yaml
        from jsonschema import validate
    except ImportError:
        print("Install: pip install -r requirements-rules.txt", file=sys.stderr)
        return 1

    spec_path = ROOT / "rules" / "rules_spec.yaml"
    schema_path = ROOT / "rules" / "rules_schema.json"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate(instance=spec, schema=schema)
    print("rules_spec.yaml is valid against rules_schema.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
