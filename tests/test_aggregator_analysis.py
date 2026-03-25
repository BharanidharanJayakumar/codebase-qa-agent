"""Tests for enhanced aggregator analysis functions."""
import os
import tempfile
from pathlib import Path
from skills.aggregator import (
    detect_architecture_pattern,
    find_entry_points,
    compute_complexity_metrics,
    detect_infrastructure,
    analyze_test_coverage,
    detect_external_dependencies,
    find_top_connected_files,
    build_comprehensive_report,
    EXT_TO_LANGUAGE,
)


def _make_file_index(files_dict):
    """Helper: build a file_index from {path: {symbols, chunks, ...}}."""
    index = {}
    for path, meta in files_dict.items():
        ext = Path(path).suffix
        index[path] = {
            "chunks": meta.get("chunks", [{"start_line": 1, "end_line": meta.get("lines", 10), "content": ""}]),
            "keywords": meta.get("keywords", []),
            "symbols": meta.get("symbols", []),
            "extension": ext,
            "size_bytes": meta.get("size_bytes", 100),
            "last_modified": 0,
        }
    return index


class TestArchitectureDetection:
    def test_mvc_pattern(self):
        fi = _make_file_index({
            "controllers/user.py": {"symbols": ["UserController"]},
            "models/user.py": {"symbols": ["User"]},
            "views/index.html": {"symbols": []},
        })
        result = detect_architecture_pattern(fi, "/tmp/fake")
        assert result["primary"]["pattern"] == "MVC (Model-View-Controller)"

    def test_layered_pattern(self):
        fi = _make_file_index({
            "services/auth.py": {"symbols": ["AuthService"]},
            "repositories/user.py": {"symbols": ["UserRepo"]},
            "controllers/api.py": {"symbols": ["ApiController"]},
        })
        result = detect_architecture_pattern(fi, "/tmp/fake")
        patterns = [d["pattern"] for d in result["all_detected"]]
        assert "Layered Architecture" in patterns

    def test_standard_fallback(self):
        fi = _make_file_index({
            "main.py": {"symbols": ["main"]},
            "utils.py": {"symbols": ["helper"]},
        })
        result = detect_architecture_pattern(fi, "/tmp/fake")
        assert result["primary"]["pattern"] == "Standard"


class TestEntryPoints:
    def test_python_main(self):
        fi = _make_file_index({
            "main.py": {"symbols": ["app", "serve"]},
            "utils.py": {"symbols": ["helper"]},
        })
        entries = find_entry_points(fi, "/tmp/fake")
        files = [e["file"] for e in entries]
        assert "main.py" in files

    def test_node_entry(self):
        fi = _make_file_index({
            "src/index.ts": {"symbols": ["bootstrap"]},
            "src/utils.ts": {"symbols": ["format"]},
        })
        entries = find_entry_points(fi, "/tmp/fake")
        files = [e["file"] for e in entries]
        assert "src/index.ts" in files


class TestComplexityMetrics:
    def test_basic_metrics(self):
        fi = _make_file_index({
            "a.py": {"symbols": ["foo", "bar"], "size_bytes": 500, "lines": 50},
            "b.py": {"symbols": ["baz"], "size_bytes": 200, "lines": 20},
        })
        metrics = compute_complexity_metrics(fi)
        assert metrics["total_files"] == 2
        assert metrics["total_symbols"] == 3
        assert metrics["total_size_bytes"] == 700
        assert len(metrics["largest_files"]) == 2
        assert len(metrics["language_breakdown"]) >= 1

    def test_language_breakdown(self):
        fi = _make_file_index({
            "app.py": {"lines": 100},
            "main.py": {"lines": 50},
            "index.js": {"lines": 200},
        })
        metrics = compute_complexity_metrics(fi)
        langs = {b["extension"] for b in metrics["language_breakdown"]}
        assert ".py" in langs
        assert ".js" in langs


class TestTestCoverage:
    def test_detects_test_files(self):
        fi = _make_file_index({
            "src/app.py": {"symbols": ["App"]},
            "tests/test_app.py": {"symbols": ["test_start"]},
        })
        result = analyze_test_coverage(fi)
        assert result["test_file_count"] == 1
        assert result["source_file_count"] == 1
        assert result["test_to_source_ratio"] == 100.0

    def test_no_tests(self):
        fi = _make_file_index({
            "src/app.py": {"symbols": ["App"]},
        })
        result = analyze_test_coverage(fi)
        assert result["test_file_count"] == 0
        assert result["test_ratio_label"] == "None detected"


class TestConnectedFiles:
    def test_fan_in(self):
        fi = _make_file_index({
            "utils.py": {"symbols": ["helper"]},
            "a.py": {"symbols": ["a"]},
            "b.py": {"symbols": ["b"]},
        })
        imports = [
            ("a.py", "utils", "utils.py"),
            ("b.py", "utils", "utils.py"),
        ]
        top = find_top_connected_files(fi, imports)
        assert len(top) >= 1
        assert top[0]["file"] == "utils.py"
        assert top[0]["imported_by_count"] == 2


class TestInfrastructure:
    def test_detects_docker(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "Dockerfile").write_text("FROM python:3.12")
            result = detect_infrastructure(d)
            assert "containerization" in result
            assert "Docker" in result["containerization"]

    def test_detects_makefile(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "Makefile").write_text("build:\n\techo ok")
            result = detect_infrastructure(d)
            assert "build" in result


class TestExternalDeps:
    def test_requirements_txt(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "requirements.txt").write_text("fastapi>=0.100\npydantic\nuvicorn")
            result = detect_external_dependencies(d)
            assert "pip" in result
            assert result["pip"]["total"] == 3
            assert "fastapi" in result["pip"]["packages"]


class TestLanguageMap:
    def test_common_extensions(self):
        assert EXT_TO_LANGUAGE["py"] == "Python"
        assert EXT_TO_LANGUAGE["ts"] == "TypeScript"
        assert EXT_TO_LANGUAGE["go"] == "Go"
        assert EXT_TO_LANGUAGE["rs"] == "Rust"


class TestComprehensiveReport:
    def test_builds_full_report(self):
        with tempfile.TemporaryDirectory() as d:
            fi = _make_file_index({
                "main.py": {"symbols": ["app"], "lines": 100},
                "tests/test_main.py": {"symbols": ["test_app"], "lines": 50},
            })
            report = build_comprehensive_report(d, fi, {"app": [{"file": "main.py", "line": 1, "type": "variable"}]})
            assert "insights" in report
            assert "complexity" in report
            assert "test_analysis" in report
            assert "architecture" in report
            assert "entry_points" in report
            assert len(report["insights"]) > 0
