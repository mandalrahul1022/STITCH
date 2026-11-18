"""Shared pytest setup for the research pilot.

Stage 1 data tests need two inputs, given as options or environment variables:
    --stitch-data DIR       (or STITCH_DATA)     a Stage 1 output directory
    --stitch-archive ZIP    (or STITCH_ARCHIVE)  STITCH_Feasibility_Results.zip

After a session that ran any test whose name contains a test ID (T01, T10b, ...), the results are
written to <stitch-data>/verification.json as rows of: id, name, status, evidence.
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

PILOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PILOT))  # the pilot modules use flat imports (from pilot import ...)

ID_PATTERN = re.compile(r"test_(T\d+[a-z]?)_")
_RESULTS = {}


def pytest_addoption(parser):
    parser.addoption("--stitch-data", default=os.environ.get("STITCH_DATA"))
    parser.addoption("--stitch-archive", default=os.environ.get("STITCH_ARCHIVE"))


def _path(config, name):
    value = config.getoption(name)
    if not value:
        pytest.fail(f"{name} (or its environment variable) is required for this test", pytrace=False)
    path = Path(value)
    if not path.exists():
        pytest.fail(f"{name} does not exist: {path}", pytrace=False)
    return path


@pytest.fixture(scope="session")
def data_dir(pytestconfig):
    return _path(pytestconfig, "--stitch-data")


@pytest.fixture(scope="session")
def archive(pytestconfig):
    return _path(pytestconfig, "--stitch-archive")


@pytest.fixture(scope="session")
def manifest(data_dir):
    return json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture
def evidence(request):
    """Tests append short human-readable facts; they are saved with the test result."""
    notes = []
    request.node.stitch_evidence = notes
    return notes


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    match = ID_PATTERN.match(item.name)
    if not match or (report.when != "call" and report.passed):
        return
    status = "PASS" if report.passed else ("SKIP" if report.skipped else "FAIL")
    notes = list(getattr(item, "stitch_evidence", []))
    if report.failed:
        notes.append(str(report.longrepr).strip().splitlines()[-1][:400])
    _RESULTS[match.group(1)] = dict(id=match.group(1), name=item.name, status=status,
                                    evidence="; ".join(notes))


def pytest_sessionfinish(session, exitstatus):
    if not _RESULTS:
        return
    value = session.config.getoption("--stitch-data")
    if not value or not Path(value).is_dir():
        return
    from tools.code_hash import code_hash  # noqa: E402

    archive = session.config.getoption("--stitch-archive")
    out = {
        "code_hash": code_hash(),
        "archive_sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest() if archive else None,
        "tests": [_RESULTS[k] for k in sorted(_RESULTS, key=lambda s: (int(re.sub(r"\D", "", s)), s))],
    }
    (Path(value) / "verification.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
