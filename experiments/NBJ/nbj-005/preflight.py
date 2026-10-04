"""Verify the approved interpreter and existing project packages before NBJ-005."""

import importlib
import json
import os
import sys
from pathlib import Path


EXPECTED_EXE = Path(
    r"C:\Users\skqja\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe"
).resolve()
SITE_PACKAGES = Path(r"C:\Kamp-2026\.venv\Lib\site-packages").resolve()
VERSIONS = {
    "numpy": "2.2.4",
    "pandas": "2.2.3",
    "scipy": "1.15.2",
    "sklearn": "1.6.1",
    "joblib": "1.4.2",
    "matplotlib": "3.10.1",
    "openpyxl": "3.1.5",
}


def main() -> None:
    packages = {}
    for name, expected_version in VERSIONS.items():
        module = importlib.import_module(name)
        location = Path(module.__file__).resolve()
        packages[name] = {
            "version": module.__version__,
            "module_file": str(location),
            "from_project_venv": location.is_relative_to(SITE_PACKAGES),
            "matches_frozen_version": module.__version__ == expected_version,
        }
    record = {
        "sys_version": sys.version,
        "sys_executable": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "packages": packages,
    }
    record["pass"] = (
        sys.version_info[:3] == (3, 13, 15)
        and Path(sys.executable).resolve() == EXPECTED_EXE
        and Path(record["pythonpath"] or "").resolve() == SITE_PACKAGES
        and all(p["from_project_venv"] and p["matches_frozen_version"] for p in packages.values())
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if not record["pass"]:
        raise SystemExit("Environment mismatch; pipeline was not started")
    target = Path(__file__).resolve().parent / "outputs" / "preflight.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
