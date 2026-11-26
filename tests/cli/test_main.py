"""End-to-end CLI behaviour with a fake model: output, modes, streams and exit codes."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from image_fixtures import make_image, write_image, write_png_with_profile
from retone.cli.__main__ import main
from retone.cli.exit_codes import ExitCode
from retone.executor import Params, load_image
from retone.inference import FakeClient, ModelUnavailableError, RaisingClient


def _run(argv: list[str], client: object) -> tuple[int, str, str]:
    out, err = StringIO(), StringIO()
    code = main(argv, client=client, out=out, err=err)  # type: ignore[arg-type]
    return code, out.getvalue(), err.getvalue()


def _photo(tmp_path: Path, name: str = "photo.png") -> Path:
    return write_image(tmp_path / name, make_image(48, 64))


def test_default_run_writes_beside_input_and_prints_path(tmp_path: Path) -> None:
    source = _photo(tmp_path)
    client = FakeClient({"exposure": 20})
    code, out, err = _run([str(source)], client)

    assert code == ExitCode.SUCCESS
    target = source.with_name("photo_retone.png")
    assert target.is_file()
    assert out.strip() == str(target)
    assert err == ""


def test_original_file_is_never_modified(tmp_path: Path) -> None:
    source = _photo(tmp_path)
    before = load_image(source)
    _run([str(source)], FakeClient({"exposure": 70}))
    assert np.array_equal(load_image(source), before)


def test_identity_model_output_reproduces_the_input(tmp_path: Path) -> None:
    source = _photo(tmp_path)
    _run([str(source)], FakeClient({}))
    rendered = load_image(source.with_name("photo_retone.png"))
    assert np.array_equal(rendered, load_image(source))


def test_params_mode_prints_without_writing(tmp_path: Path) -> None:
    source = _photo(tmp_path)
    code, out, err = _run([str(source), "--params"], FakeClient({"exposure": 12, "contrast": -4}))
    assert code == ExitCode.SUCCESS
    assert not source.with_name("photo_retone.png").exists()
    assert "exposure" in out and "+12" in out
    assert "contrast" in out and "-4" in out
    assert err == ""


def test_json_mode_prints_parameters_only(tmp_path: Path) -> None:
    source = _photo(tmp_path)
    code, out, err = _run([str(source), "--json"], FakeClient({"exposure": 12}))
    assert code == ExitCode.SUCCESS
    assert not source.with_name("photo_retone.png").exists()
    assert json.loads(out) == Params(exposure=12).as_dict()
    assert err == ""


def test_explicit_params_skip_the_model(tmp_path: Path) -> None:
    source = _photo(tmp_path)
    client = RaisingClient(ModelUnavailableError("should not be called"))
    code, out, _ = _run([str(source), "--set", "exposure=15", "--set", "contrast=-3"], client)
    assert code == ExitCode.SUCCESS
    assert source.with_name("photo_retone.png").is_file()
    assert out.strip().endswith("photo_retone.png")


def test_out_path_is_honoured(tmp_path: Path) -> None:
    source = _photo(tmp_path)
    target = tmp_path / "elsewhere.png"
    code, out, _ = _run([str(source), "--out", str(target)], FakeClient({}))
    assert code == ExitCode.SUCCESS
    assert target.is_file()
    assert out.strip() == str(target)


def test_multiple_inputs_are_all_processed(tmp_path: Path) -> None:
    first = _photo(tmp_path, "one.png")
    second = _photo(tmp_path, "two.png")
    code, out, _ = _run([str(first), str(second)], FakeClient({"exposure": 5}))
    assert code == ExitCode.SUCCESS
    assert first.with_name("one_retone.png").is_file()
    assert second.with_name("two_retone.png").is_file()
    assert len(out.strip().splitlines()) == 2


def test_out_with_multiple_inputs_is_a_usage_error(tmp_path: Path) -> None:
    first = _photo(tmp_path, "one.png")
    second = _photo(tmp_path, "two.png")
    code, _, err = _run([str(first), str(second), "--out", str(tmp_path / "x.png")], FakeClient({}))
    assert code == ExitCode.USAGE
    assert "--out" in err


def test_existing_output_requires_force(tmp_path: Path) -> None:
    source = _photo(tmp_path)
    target = source.with_name("photo_retone.png")
    write_image(target, make_image(48, 64, seed=9))
    code, _, err = _run([str(source)], FakeClient({}))
    assert code == ExitCode.WRITE
    assert "--force" in err

    code, _, _ = _run([str(source), "--force"], FakeClient({}))
    assert code == ExitCode.SUCCESS


def test_missing_input_is_an_input_error(tmp_path: Path) -> None:
    code, _, err = _run([str(tmp_path / "absent.png")], FakeClient({}))
    assert code == ExitCode.INPUT
    assert "absent.png" in err


def test_model_unavailable_is_reported_cleanly(tmp_path: Path) -> None:
    source = _photo(tmp_path)
    code, _, err = _run([str(source)], RaisingClient(ModelUnavailableError("server down")))
    assert code == ExitCode.MODEL
    assert "server down" in err
    assert "Traceback" not in err


def test_clamping_warns_on_stderr_but_still_writes(tmp_path: Path) -> None:
    source = _photo(tmp_path)
    code, out, err = _run([str(source)], FakeClient({"exposure": 500}))
    assert code == ExitCode.SUCCESS
    assert source.with_name("photo_retone.png").is_file()
    assert "clamped" in err
    assert out.strip() == str(source.with_name("photo_retone.png"))


def test_invalid_output_falls_back_and_warns(tmp_path: Path) -> None:
    source = _photo(tmp_path)
    code, _, err = _run([str(source)], FakeClient({"brightness": 5}))
    assert code == ExitCode.SUCCESS
    assert "defaults" in err


@pytest.mark.parametrize("style", ["side-by-side", "split"])
def test_compare_writes_a_comparison_image(tmp_path: Path, style: str) -> None:
    source = _photo(tmp_path)
    code, _, _ = _run([str(source), "--compare", style], FakeClient({"exposure": 10}))
    assert code == ExitCode.SUCCESS
    tag = "compare" if style == "side-by-side" else "split"
    assert source.with_name(f"photo_retone_{tag}.png").is_file()


def test_compare_both_writes_two_images(tmp_path: Path) -> None:
    source = _photo(tmp_path)
    code, _, _ = _run([str(source), "--compare", "both"], FakeClient({"exposure": 10}))
    assert code == ExitCode.SUCCESS
    assert source.with_name("photo_retone_compare.png").is_file()
    assert source.with_name("photo_retone_split.png").is_file()


def test_non_srgb_profile_warns_on_stderr(tmp_path: Path) -> None:
    source = write_png_with_profile(tmp_path / "wide.png", make_image(48, 64), "Adobe RGB (1998)")
    code, _, err = _run([str(source), "--set", "exposure=5"], FakeClient({}))
    assert code == ExitCode.SUCCESS
    assert "non-sRGB ICC profile" in err
