"""Tests for the `python -m src.main modality` subcommand.

Tests exercise the cmd_modality function directly by constructing an
argparse.Namespace; this avoids invoking argparse + the full CLI dispatch
and keeps tests offline-clean.
"""

import argparse
import json

import pytest

from src.main import cmd_modality


def _args(**overrides) -> argparse.Namespace:
    """Build a Namespace with sensible defaults."""
    base = {
        "pdf_path": "",
        "output": None,
        "with_classifier": False,
        "with_polarity": False,
        "show": 5,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cmd_modality_raw_text_default(tmp_path, capsys):
    """Default path: .txt file, no classifier, prints summary + findings."""
    contract = tmp_path / "small.txt"
    contract.write_text(
        "Licensee shall not reverse engineer the Software.\n"
        "Licensor may, at its sole option, terminate.\n"
        "Either party shall comply with applicable law."
    )
    cmd_modality(_args(pdf_path=str(contract)))

    out = capsys.readouterr().out
    assert "MODALITY ANALYSIS" in out
    assert "small.txt" in out
    assert "PROHIBITION" in out
    assert "OBLIGATION" in out
    # show=5, sample contract has at least 3 findings
    assert "Top" in out


def test_cmd_modality_show_zero_skips_findings_list(tmp_path, capsys):
    contract = tmp_path / "x.txt"
    contract.write_text("Licensee shall pay all fees.")
    cmd_modality(_args(pdf_path=str(contract), show=0))

    out = capsys.readouterr().out
    assert "MODALITY ANALYSIS" in out
    assert "OBLIGATION" in out
    # No "Top N findings:" header when show=0
    assert "Top " not in out


def test_cmd_modality_writes_json_output(tmp_path):
    contract = tmp_path / "x.txt"
    contract.write_text(
        "Licensee shall not reverse engineer.\n"
        "Either party may terminate."
    )
    out_path = tmp_path / "report.json"
    cmd_modality(_args(pdf_path=str(contract), output=str(out_path), show=0))

    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    # Default path serializes the ModalityReport directly (no wrapping key).
    assert "contract_id" in payload
    assert "findings" in payload
    assert "counts" in payload
    assert payload["counts"]["PROHIBITION"] >= 1
    assert payload["counts"]["PERMISSION"] >= 1


def test_cmd_modality_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cmd_modality(_args(pdf_path=str(tmp_path / "does_not_exist.txt")))
    assert exc.value.code == 1


def test_cmd_modality_finding_counts_match_terminal_summary(tmp_path, capsys):
    """The per-modality counts printed should match the actual finding rate."""
    contract = tmp_path / "x.txt"
    # 2 prohibitions, 1 obligation, 1 permission
    contract.write_text(
        "Licensee shall not assign. Licensor must not disclose. "
        "Vendor shall maintain insurance. Either party may terminate."
    )
    cmd_modality(_args(pdf_path=str(contract), show=0))

    out = capsys.readouterr().out
    assert "PROHIBITION 2" in out
    assert "OBLIGATION  1" in out
    assert "PERMISSION  1" in out
