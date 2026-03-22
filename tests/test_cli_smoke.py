"""Smoke tests for landmarkdiff CLI subcommands."""

import subprocess


def test_main_help_returns_zero():
    result = subprocess.run(
        ["python", "-m", "landmarkdiff", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_main_help_shows_subcommands():
    result = subprocess.run(
        ["python", "-m", "landmarkdiff", "--help"],
        capture_output=True,
        text=True,
    )
    assert "infer" in result.stdout
    assert "landmarks" in result.stdout
    assert "demo" in result.stdout


def test_infer_help_returns_zero():
    result = subprocess.run(
        ["python", "-m", "landmarkdiff", "infer", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_infer_help_shows_flags():
    result = subprocess.run(
        ["python", "-m", "landmarkdiff", "infer", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--procedure" in result.stdout
    assert "--intensity" in result.stdout


def test_landmarks_help_returns_zero():
    result = subprocess.run(
        ["python", "-m", "landmarkdiff", "landmarks", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_landmarks_help_shows_flags():
    result = subprocess.run(
        ["python", "-m", "landmarkdiff", "landmarks", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--output" in result.stdout


def test_demo_help_returns_zero():
    result = subprocess.run(
        ["python", "-m", "landmarkdiff", "demo", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
