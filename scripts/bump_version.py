#!/usr/bin/env python3
"""Bump the version across all files that reference it.

Updates version in:
- pyproject.toml
- landmarkdiff/__init__.py
- model_card.yaml
- Dockerfile

Usage:
    python scripts/bump_version.py patch    # 0.3.0 -> 0.3.1
    python scripts/bump_version.py minor    # 0.3.0 -> 0.4.0
    python scripts/bump_version.py major    # 0.3.0 -> 1.0.0
    python scripts/bump_version.py 0.4.0    # explicit version
    python scripts/bump_version.py --show   # show current version
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VERSION_FILES = [
    ("pyproject.toml", r'version\s*=\s*"(\d+\.\d+\.\d+)"'),
    ("landmarkdiff/__init__.py", r'__version__\s*=\s*"(\d+\.\d+\.\d+)"'),
    ("model_card.yaml", r'version:\s*"(\d+\.\d+\.\d+)"'),
    ("Dockerfile", r'version="(\d+\.\d+\.\d+)"'),
]


def get_current_version() -> str:
    """Read current version from pyproject.toml."""
    content = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'version\s*=\s*"(\d+\.\d+\.\d+)"', content)
    if not match:
        raise ValueError("Cannot find version in pyproject.toml")
    return match.group(1)


def compute_new_version(current: str, bump: str) -> str:
    """Compute new version based on bump type."""
    if re.match(r"\d+\.\d+\.\d+$", bump):
        return bump

    major, minor, patch = (int(x) for x in current.split("."))

    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    elif bump == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump == "major":
        return f"{major + 1}.0.0"
    else:
        raise ValueError(f"Unknown bump type: {bump}. Use: patch, minor, major, or X.Y.Z")


def update_file(filepath: str, pattern: str, old_version: str, new_version: str) -> bool:
    """Update version in a single file. Returns True if file was updated."""
    full_path = ROOT / filepath
    if not full_path.exists():
        return False

    content = full_path.read_text()
    new_content = content.replace(old_version, new_version)

    if new_content != content:
        full_path.write_text(new_content)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Bump LandmarkDiff version")
    parser.add_argument(
        "bump", nargs="?", default=None, help="Bump type: patch, minor, major, or explicit X.Y.Z"
    )
    parser.add_argument("--show", action="store_true", help="Show current version")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change")
    args = parser.parse_args()

    current = get_current_version()

    if args.show or args.bump is None:
        print(f"Current version: {current}")
        if args.bump is None and not args.show:
            parser.print_help()
        return

    new = compute_new_version(current, args.bump)
    print(f"Bumping version: {current} -> {new}")

    if args.dry_run:
        print("\nDry run -- no files modified:")
        for filepath, _pattern in VERSION_FILES:
            full_path = ROOT / filepath
            if full_path.exists():
                content = full_path.read_text()
                if current in content:
                    print(f"  Would update: {filepath}")
                else:
                    print(f"  No match: {filepath}")
            else:
                print(f"  Not found: {filepath}")
        return

    updated = []
    for filepath, pattern in VERSION_FILES:
        if update_file(filepath, pattern, current, new):
            updated.append(filepath)
            print(f"  Updated: {filepath}")
        else:
            full_path = ROOT / filepath
            if full_path.exists():
                print(f"  Skipped: {filepath} (no match)")
            else:
                print(f"  Missing: {filepath}")

    if updated:
        print(f"\nVersion bumped to {new} in {len(updated)} files")
        print("\nNext steps:")
        print(f"  git add {' '.join(updated)}")
        print(f'  git commit -m "chore: bump version to {new}"')
    else:
        print("No files updated")


if __name__ == "__main__":
    main()
