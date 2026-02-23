"""Tests for utility functions."""

import os
import json
import pytest
import tempfile
from pathlib import Path

from textSummarizer.utils.common import (
    read_yaml,
    create_directories,
    get_size,
    save_json,
    load_json,
)


class TestReadYaml:
    """Tests for read_yaml utility."""

    def test_read_valid_yaml(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key1: value1\nkey2: 42\n")
        result = read_yaml(yaml_file)
        assert result.key1 == "value1"
        assert result.key2 == 42

    def test_read_empty_yaml(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        with pytest.raises(ValueError, match="empty"):
            read_yaml(yaml_file)

    def test_read_nonexistent_yaml(self):
        with pytest.raises(FileNotFoundError):
            read_yaml(Path("nonexistent.yaml"))

    def test_read_nested_yaml(self, tmp_path):
        yaml_file = tmp_path / "nested.yaml"
        yaml_file.write_text("parent:\n  child: value\n  number: 10\n")
        result = read_yaml(yaml_file)
        assert result.parent.child == "value"
        assert result.parent.number == 10


class TestCreateDirectories:
    """Tests for create_directories utility."""

    def test_create_single_dir(self, tmp_path):
        new_dir = str(tmp_path / "new_dir")
        create_directories([new_dir], verbose=False)
        assert os.path.exists(new_dir)

    def test_create_multiple_dirs(self, tmp_path):
        dirs = [str(tmp_path / f"dir_{i}") for i in range(3)]
        create_directories(dirs, verbose=False)
        for d in dirs:
            assert os.path.exists(d)

    def test_create_existing_dir(self, tmp_path):
        # Should not raise
        create_directories([str(tmp_path)], verbose=False)


class TestGetSize:
    """Tests for get_size utility."""

    def test_small_file(self, tmp_path):
        f = tmp_path / "small.txt"
        f.write_text("hello")
        size = get_size(f)
        assert "B" in size

    def test_larger_file(self, tmp_path):
        f = tmp_path / "larger.txt"
        f.write_text("x" * 2048)
        size = get_size(f)
        assert "KB" in size


class TestJsonUtils:
    """Tests for save_json and load_json utilities."""

    def test_save_and_load_json(self, tmp_path):
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        json_path = tmp_path / "test.json"
        save_json(json_path, data)
        loaded = load_json(json_path)
        assert loaded == data

    def test_save_json_creates_parent_dirs(self, tmp_path):
        json_path = tmp_path / "sub" / "dir" / "test.json"
        save_json(json_path, {"key": "value"})
        assert json_path.exists()
