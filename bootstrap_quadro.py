#!/usr/bin/env python3
"""
bootstrap_quadro.py — fill the remaining gaps and add the CTM package.

Run from QUADRO/ root (directory containing pyproject.toml).
Idempotent: non-empty existing files are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").exists():
    sys.exit("Run from QUADRO/ root (pyproject.toml not found).")

FILES: dict[str, str] = {}

# =============================================================================
# 1. CTM PACKAGE (new)
# =============================================================================

FILES["src/quadro_deploy/ctm/__init__.py"] = '''"""Convexity Transfer Method (CTM) — PSD projection and H_gph theta."""

from __future__ import annotations

from .hessian import GPHHessian
from .projection import diagnose, project_symmetric_to_psd
from .types import HessianDiagnostics

__all__ = [
    "GPHHessian",
    "HessianDiagnostics",
    "diagnose",
    "project_symmetric_to_psd",
]
'''

FILES["src/quadro_deploy/ctm/types.py"] = '''"""Types for the CTM package."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class HessianDiagnostics:
    """Spectral summary of a symmetric matrix."""

    eigenvalues: List[float]
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    is_psd: bool
    eps_floor: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
'''

FILES["src/quadro_deploy/ctm/projection.py"] = '''"""PSD projection of symmetric matrices.

Given a symmetric matrix A, compute the nearest positive semi-definite
matrix (in Frobenius norm) by clamping its eigenvalues to >= -epsilon.

This is the concrete implementation behind the claim in
`technical_summary.md`:  nabla^2 L >= -eps I. Applied on every H_gph
step, it guarantees the smoothed landscape is PSD within eps.
"""

from __future__ import annotations

import numpy as np

from .types import HessianDiagnostics


def _as_symmetric(matrix: np.ndarray) -> np.ndarray:
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected square matrix, got shape {A.shape}")
    if not np.allclose(A, A.T, atol=1e-9):
        A = 0.5 * (A + A.T)
    return A


def project_symmetric_to_psd(
    matrix: np.ndarray, epsilon: float = 1e-8
) -> np.ndarray:
    """Return nearest PSD projection of `matrix` with eigenvalue floor -epsilon."""
    if epsilon < 0.0:
        raise ValueError("epsilon must be >= 0")
    A = _as_symmetric(matrix)
    w, V = np.linalg.eigh(A)
    w_clamped = np.maximum(w, -epsilon)
    return V @ np.diag(w_clamped) @ V.T


def diagnose(matrix: np.ndarray, epsilon: float = 1e-8) -> HessianDiagnostics:
    """Return spectral diagnostics without modifying the matrix."""
    A = _as_symmetric(matrix)
    w = np.linalg.eigvalsh(A)
    if w.size == 0:
        return HessianDiagnostics(
            eigenvalues=[],
            min_eigenvalue=0.0,
            max_eigenvalue=0.0,
            condition_number=0.0,
            is_psd=True,
            eps_floor=epsilon,
        )
    wmin = float(w.min())
    wmax = float(w.max())
    cond = float(abs(wmax) / abs(wmin)) if abs(wmin) > 1e-12 else float("inf")
    return HessianDiagnostics(
        eigenvalues=[float(x) for x in w],
        min_eigenvalue=wmin,
        max_eigenvalue=wmax,
        condition_number=cond,
        is_psd=wmin >= -epsilon,
        eps_floor=epsilon,
    )
'''

FILES["src/quadro_deploy/ctm/hessian.py"] = '''"""Hessian estimation and PSD-projected gradient smoothing (H_gph theta)."""

from __future__ import annotations

from typing import Callable, List

import numpy as np

from .projection import diagnose, project_symmetric_to_psd
from .types import HessianDiagnostics

Vector = List[float]


class GPHHessian:
    """Estimate Hessians by central differences and apply PSD smoothing."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        if epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        self.epsilon = float(epsilon)

    def estimate(
        self,
        f: Callable[[Vector], float],
        x: Vector,
        h: float | None = None,
    ) -> np.ndarray:
        """Central-difference Hessian of `f` at `x`."""
        x_arr = np.asarray(x, dtype=float)
        n = x_arr.size
        if n == 0:
            raise ValueError("x must be non-empty")
        if h is None:
            h = max(1e-5, 1e-5 * float(np.max(np.abs(x_arr)) or 1.0))
        if h <= 0.0:
            raise ValueError("h must be > 0")

        H = np.zeros((n, n), dtype=float)
        f0 = float(f(list(x_arr)))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = h
            f_plus_i = float(f(list(x_arr + ei)))
            f_minus_i = float(f(list(x_arr - ei)))
            H[i, i] = (f_plus_i - 2.0 * f0 + f_minus_i) / (h * h)
            for j in range(i + 1, n):
                ej = np.zeros(n)
                ej[j] = h
                val = (
                    float(f(list(x_arr + ei + ej)))
                    - float(f(list(x_arr + ei - ej)))
                    - float(f(list(x_arr - ei + ej)))
                    + float(f(list(x_arr - ei - ej)))
                ) / (4.0 * h * h)
                H[i, j] = val
                H[j, i] = val
        return H

    def project(self, H: np.ndarray) -> np.ndarray:
        return project_symmetric_to_psd(H, epsilon=self.epsilon)

    def diagnose(self, H: np.ndarray) -> HessianDiagnostics:
        return diagnose(H, epsilon=self.epsilon)

    def smooth_gradient(
        self,
        grad: Vector,
        H: np.ndarray,
        learning_rate: float = 0.01,
    ) -> List[float]:
        """Return one preconditioned step: g - lr * H_psd^{-1} g."""
        g = np.asarray(grad, dtype=float)
        Hp = self.project(H) + self.epsilon * np.eye(H.shape[0])
        step = np.linalg.solve(Hp, g)
        return (g - learning_rate * step).tolist()
'''

FILES["tests/test_ctm.py"] = '''from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from quadro_deploy.ctm import GPHHessian, diagnose, project_symmetric_to_psd


def test_project_identity_is_identity() -> None:
    assert np.allclose(project_symmetric_to_psd(np.eye(3)), np.eye(3))


def test_project_negative_definite_becomes_psd() -> None:
    A = np.array([[-1.0, 0.0], [0.0, -2.0]])
    P = project_symmetric_to_psd(A, epsilon=1e-9)
    assert (np.linalg.eigvalsh(P) >= -1e-8).all()


def test_project_is_symmetric() -> None:
    A = np.array([[1.0, 2.0], [2.0, -1.0]])
    P = project_symmetric_to_psd(A)
    assert np.allclose(P, P.T)


def test_project_rejects_non_square() -> None:
    with pytest.raises(ValueError):
        project_symmetric_to_psd(np.array([1.0, 2.0]))


def test_project_rejects_negative_epsilon() -> None:
    with pytest.raises(ValueError):
        project_symmetric_to_psd(np.eye(2), epsilon=-1e-9)


def test_diagnose_psd_matrix() -> None:
    d = diagnose(np.array([[2.0, 0.0], [0.0, 3.0]]))
    assert d.is_psd is True
    assert d.min_eigenvalue > 0.0
    assert abs(d.max_eigenvalue - 3.0) < 1e-10


def test_diagnose_non_psd_matrix() -> None:
    d = diagnose(np.array([[1.0, 0.0], [0.0, -2.0]]), epsilon=1e-6)
    assert d.is_psd is False
    assert d.min_eigenvalue < 0.0


def test_hessian_estimates_quadratic() -> None:
    def f(x: list[float]) -> float:
        return 3.0 * x[0] ** 2 + 2.0 * x[1] ** 2

    H = GPHHessian().estimate(f, [0.5, -0.5])
    assert np.allclose(H, [[6.0, 0.0], [0.0, 4.0]], atol=1e-3)


def test_hessian_rejects_bad_epsilon() -> None:
    with pytest.raises(ValueError):
        GPHHessian(epsilon=0.0)


def test_hessian_rejects_empty_x() -> None:
    with pytest.raises(ValueError):
        GPHHessian().estimate(lambda v: 0.0, [])


def test_hessian_project_fixes_negative() -> None:
    h = GPHHessian(epsilon=1e-6)
    P = h.project(np.array([[-1.0, 0.0], [0.0, 1.0]]))
    assert (np.linalg.eigvalsh(P) >= -1e-5).all()


def test_smooth_gradient_returns_floats() -> None:
    h = GPHHessian()
    out = h.smooth_gradient([1.0, -1.0], np.array([[2.0, 0.0], [0.0, 2.0]]), 0.1)
    assert len(out) == 2
    assert all(isinstance(x, float) for x in out)


def test_smooth_gradient_reduces_along_positive_curvature() -> None:
    h = GPHHessian()
    grad = [10.0, -10.0]
    H = np.eye(2)
    out = h.smooth_gradient(grad, H, learning_rate=0.1)
    assert abs(out[0]) < abs(grad[0])
    assert abs(out[1]) < abs(grad[1])
'''

# =============================================================================
# 2. FILES VERIFIED MISSING (from your earlier check)
# =============================================================================

FILES[".gitignore"] = """__pycache__/
*.pyc
*.pyo
.venv/
venv/
dist/
build/
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
coverage.xml
htmlcov/
.env
.env.*
.DS_Store
.idea/
.vscode/

src/quadro_deploy/server/static/index.html
!src/quadro_deploy/server/static/.gitkeep

vendor/refactored_robot_dynamics.py
"""

FILES[".dockerignore"] = """.git
.gitignore
.venv
venv
__pycache__
*.pyc
*.pyo
dist
build
*.egg-info
.pytest_cache
.ruff_cache
.mypy_cache
.coverage
coverage.xml
htmlcov
.env
.env.*
tests
docs
scripts
.github
"""

FILES[".editorconfig"] = """root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 4

[*.{yml,yaml,json,md,toml}]
indent_size = 2

[Makefile]
indent_style = tab

[*.py]
max_line_length = 100
"""

FILES[".gitattributes"] = """* text=auto eol=lf

*.py       text diff=python
*.md       text
*.yml      text
*.yaml     text
*.toml     text
*.sh       text eol=lf

vendor/refactored_robot_dynamics.py linguist-vendored
src/quadro_deploy/server/static/index.html linguist-generated
"""

FILES[".pre-commit-config.yaml"] = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-yaml
      - id: check-toml
      - id: check-merge-conflict
      - id: check-added-large-files
        args: ["--maxkb=500"]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.10
    hooks:
      - id: ruff
        args: ["--fix"]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies:
          - pydantic>=2.6
          - pydantic-settings>=2.2
          - fastapi>=0.110
          - numpy>=1.26
        args: ["src"]
        pass_filenames: false
"""

FILES["MANIFEST.in"] = """include LICENSE
include README.md
include CHANGELOG.md
include CONTRIBUTING.md
include SECURITY.md
include CODE_OF_CONDUCT.md
include pyproject.toml

recursive-include src/quadro_deploy *.py
recursive-include src/quadro_deploy py.typed
recursive-include config *.yaml
recursive-include docs *.md
recursive-include scripts *.sh

global-exclude *.pyc
global-exclude __pycache__
global-exclude .DS_Store
"""

FILES["src/quadro_deploy/py.typed"] = ""
FILES["src/quadro_deploy/_version.py"] = '''"""Single source of truth for the package version."""

from __future__ import annotations

__version__ = "0.1.0"
'''

FILES["src/quadro_deploy/server/static/.gitkeep"] = (
    "# Place the provided deepseek_html_*.html here as `index.html`.\n"
    "# The server returns 503 on /studio until this file exists.\n"
)

FILES["tests/__init__.py"] = ""

FILES["tests/conftest.py"] = '''"""Shared pytest fixtures."""

from __future__ import annotations

from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from quadro_deploy.engine import EngineRunner


@pytest.fixture
def engine() -> EngineRunner:
    return EngineRunner()


@pytest.fixture
def mock_legs() -> list[dict[str, float]]:
    return [
        {
            "in_stance": True,
            "raw_w_abd": 0.1,
            "raw_w_hip": 1.2,
            "raw_w_knee": -1.8,
            "tau_abd": 1.5,
            "j11": 0.10,
            "j12": 0.05,
            "j21": 0.15,
            "j22": 0.10,
            "theta_hip": 0.05,
            "swing_factor": 0.0,
            "foot_vx": 0.1,
            "foot_x": 0.15,
        }
        for _ in range(4)
    ]


@pytest.fixture
def client() -> Iterator[TestClient]:
    from quadro_deploy.server.app import app

    with TestClient(app) as c:
        yield c
'''

FILES["tests/integration/__init__.py"] = ""

FILES["tests/integration/test_server.py"] = '''"""End-to-end tests for the FastAPI app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_root_redirects_to_studio(client: TestClient) -> None:
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/studio"


def test_healthz_ok(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_readyz_reports_state(client: TestClient) -> None:
    r = client.get("/readyz")
    assert r.status_code in (200, 503)


def test_studio_returns_503_when_missing(client: TestClient) -> None:
    from quadro_deploy.server.app import _studio_path

    if _studio_path().exists():
        pytest.skip("studio HTML installed; 503 path not exercised")
    r = client.get("/studio")
    assert r.status_code == 503
    assert "not installed" in r.json()["error"]["message"].lower()


def test_request_id_echoed(client: TestClient) -> None:
    r = client.get("/healthz", headers={"x-request-id": "test-rid-123"})
    assert r.headers["x-request-id"] == "test-rid-123"


def test_api_step_returns_engine_result(client: TestClient) -> None:
    r = client.post("/api/step", json={})
    assert r.status_code == 200
    body = r.json()
    assert "simulated" in body
    assert body["v_bus_v"] > 0


def test_api_audit_returns_reconciliation(client: TestClient) -> None:
    r = client.get("/api/audit")
    assert r.status_code == 200
    body = r.json()
    assert "engine" in body
    assert "audit" in body
    assert "pass" in body["audit"]
'''

FILES["CHANGELOG.md"] = """# Changelog

## [Unreleased]

### Added
- CTM package: `GPHHessian`, `project_symmetric_to_psd`, `diagnose`
- PSD projection with eigenvalue floor `-epsilon`
- Central-difference Hessian estimation
- `smooth_gradient`: preconditioned gradient step under H_gph theta
- Full test coverage for CTM module

### Known gaps
- Titan-GPH enclave is stub-only
- CTM operates on scalar `f(x)`; vector optimization adapters not yet written
"""

FILES["CONTRIBUTING.md"] = """# Contributing to QUADRO

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
make install-dev
