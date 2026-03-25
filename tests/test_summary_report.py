"""Tests for the enhanced summary report endpoint logic."""
import tempfile
from pathlib import Path
from skills.aggregator import (
    build_comprehensive_report,
    _generate_insights,
    _human_size,
)


class TestHumanSize:
    def test_bytes(self):
        assert _human_size(500) == "500 B"

    def test_kilobytes(self):
        assert _human_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert _human_size(1048576) == "1.0 MB"

    def test_zero(self):
        assert _human_size(0) == "0 B"


class TestInsights:
    def test_small_project(self):
        insights = _generate_insights(
            summary={"framework_hints": ["FastAPI"]},
            architecture={"primary": {"pattern": "Standard", "description": ""}},
            complexity={"total_files": 5, "total_lines": 200, "avg_file_size_lines": 40, "most_complex_files": []},
            tests={"test_to_source_ratio": 50.0, "test_ratio_label": "Good", "test_file_count": 2},
            infrastructure={},
            entry_points=[{"file": "main.py"}],
        )
        assert any("Small project" in i for i in insights)
        assert any("FastAPI" in i for i in insights)
        assert any("Good" in i for i in insights)
        assert any("main.py" in i for i in insights)

    def test_large_project(self):
        insights = _generate_insights(
            summary={"framework_hints": []},
            architecture={"primary": {"pattern": "Layered Architecture", "description": "Organized in layers"}},
            complexity={"total_files": 250, "total_lines": 50000, "avg_file_size_lines": 200, "most_complex_files": [{"file": "big.py", "symbol_count": 50}]},
            tests={"test_to_source_ratio": 0, "test_ratio_label": "None detected", "test_file_count": 0},
            infrastructure={"ci_cd": ["GitHub Actions"], "containerization": ["Docker"]},
            entry_points=[],
        )
        assert any("Large codebase" in i for i in insights)
        assert any("Layered" in i for i in insights)
        assert any("No test files" in i for i in insights)
        assert any("GitHub Actions" in i for i in insights)
        assert any("Docker" in i for i in insights)


class TestComprehensiveReport:
    def test_report_structure(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "requirements.txt").write_text("fastapi\nuvicorn")
            fi = {
                "app.py": {
                    "chunks": [{"start_line": 1, "end_line": 50, "content": "from fastapi import FastAPI"}],
                    "keywords": ["fastapi", "app"],
                    "symbols": ["app", "create_app"],
                    "extension": ".py",
                    "size_bytes": 1200,
                    "last_modified": 0,
                },
                "tests/test_app.py": {
                    "chunks": [{"start_line": 1, "end_line": 30, "content": "import pytest"}],
                    "keywords": ["test", "pytest"],
                    "symbols": ["test_startup"],
                    "extension": ".py",
                    "size_bytes": 400,
                    "last_modified": 0,
                },
            }
            sm = {"app": [{"file": "app.py", "line": 5, "type": "variable"}]}
            report = build_comprehensive_report(d, fi, sm)

            assert "insights" in report
            assert "complexity" in report
            assert "test_analysis" in report
            assert "external_dependencies" in report
            assert report["complexity"]["total_files"] == 2
            assert report["test_analysis"]["test_file_count"] == 1
            assert "pip" in report["external_dependencies"]
