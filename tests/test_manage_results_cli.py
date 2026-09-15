from __future__ import annotations

from pathlib import Path

import pytest


def test_manage_results_parser_requires_promote_arguments():
    from scripts.manage_results import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["promote"])


def test_manage_results_parser_defaults_release_root_and_dirty_flag():
    from scripts.manage_results import build_parser

    args = build_parser().parse_args([
        "promote", "--run", "some-run", "--release", "baseline_v1"
    ])
    assert args.run == Path("some-run")
    assert args.release == "baseline_v1"
    assert args.releases_root == Path("outputs") / "releases"
    assert args.allow_dirty is False

    dirty_args = build_parser().parse_args([
        "promote", "--run", "some-run", "--release", "baseline_v1", "--allow-dirty"
    ])
    assert dirty_args.allow_dirty is True
