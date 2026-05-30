# AVL Tree - Nick Thompson
# An implementation of a self-balancing binary search tree using AVL algorithm
#
# Copyright (c) 2026 Nick Thompson
# SPDX-License-Identifier: MIT

"""AVL Tree functional tests."""
import subprocess
import pytest


@pytest.fixture(scope="session")
def avl_tree_binary(tmp_path_factory):
    """Build the avl_tree_app binary before any test runs.
    
    Uses cmake to build the target from project root. Fails the session if build fails.
    """
    build_dir = "build"
    result = subprocess.run(
        ["cmake", "--build", build_dir, "--target", "avl_tree_app"],
        cwd=".",
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"Build failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return f"{build_dir}/app/avl_tree_app"


def test_returns_zero(avl_tree_binary):
    """Test that invoking the binary returns exit code 0."""
    result = subprocess.run(
        [avl_tree_binary],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_first_five_lines_sorted(avl_tree_binary):
    """Test that first 5 stdout lines are sorted key-value pairs."""
    result = subprocess.run(
        [avl_tree_binary],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = result.stdout.strip().split("\n")
    first_five = lines[:5]

    keys = []
    for line in first_five:
        assert ": " in line, f"Line does not contain 'key: value' format: {line}"
        key, _ = line.split(": ", 1)
        keys.append(key)

    assert keys == sorted(keys), f"Keys are not alphabetically sorted: {keys}"


def test_search_cherry_exact(avl_tree_binary):
    """Test that one stdout line equals exactly 'search cherry: 1'."""
    result = subprocess.run(
        [avl_tree_binary],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = result.stdout.strip().split("\n")
    assert "search cherry: 1" in lines


def test_search_mango_exact(avl_tree_binary):
    """Test that one stdout line equals exactly 'search mango: not found'."""
    result = subprocess.run(
        [avl_tree_binary],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = result.stdout.strip().split("\n")
    assert "search mango: not found" in lines


def test_height_within_bound(avl_tree_binary):
    """Test that one stdout line begins with 'height: ' and integer is <= 3."""
    result = subprocess.run(
        [avl_tree_binary],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = result.stdout.strip().split("\n")
    height_line = None
    for line in lines:
        if line.startswith("height: "):
            height_line = line
            break

    assert height_line is not None, f"No 'height: <value>' line found in output:\n{result.stdout}"

    try:
        height_value = int(height_line.split(": ")[1])
    except (ValueError, IndexError) as e:
        pytest.fail(f"Could not parse integer from height line '{height_line}': {e}")

    assert height_value <= 3, f"Height {height_value} exceeds bound of 3"
