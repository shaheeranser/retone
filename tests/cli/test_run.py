"""The library pipeline: explicit params skip the model, and output safety rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from image_fixtures import make_image, write_image
from retone.cli.errors import CliError
from retone.cli.exit_codes import ExitCode
from retone.cli.run import choose_params, enhance_image, resolve_output
from retone.executor import Params, apply
from retone.inference import FakeClient, ModelUnavailableError, RaisingClient


def test_explicit_params_never_call_the_client() -> None:
    client = RaisingClient(ModelUnavailableError("boom"))
    params, warnings = choose_params(make_image(), client, explicit=Params(exposure=9))
    assert params == Params(exposure=9)
    assert warnings == ()


def test_model_unavailable_becomes_a_model_cli_error() -> None:
    with pytest.raises(CliError) as raised:
        choose_params(make_image(), RaisingClient(ModelUnavailableError("down")))
    assert raised.value.code == ExitCode.MODEL


def test_enhance_image_applies_the_chosen_params() -> None:
    image = make_image(40, 50)
    rendered, params, _ = enhance_image(image, FakeClient({"exposure": 20}))
    assert params == Params(exposure=20)
    assert np.array_equal(rendered, apply(image, Params(exposure=20)))
    assert not np.array_equal(rendered, image)


def test_default_output_refuses_to_equal_the_input(tmp_path: Path) -> None:
    source = write_image(tmp_path / "photo.png", make_image())
    with pytest.raises(CliError) as raised:
        resolve_output(source, source, overwrite=False)
    assert raised.value.code == ExitCode.USAGE


def test_existing_output_without_force_is_a_write_error(tmp_path: Path) -> None:
    source = write_image(tmp_path / "photo.png", make_image())
    existing = write_image(tmp_path / "photo_retone.png", make_image())
    with pytest.raises(CliError) as raised:
        resolve_output(source, None, overwrite=False)
    assert raised.value.code == ExitCode.WRITE
    assert str(existing) in raised.value.message
    assert "--force" in raised.value.message
