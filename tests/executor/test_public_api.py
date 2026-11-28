from __future__ import annotations

import retone.executor as executor


def test_public_api_is_exactly_what_the_package_exports() -> None:
    assert sorted(executor.__all__) == sorted(
        [
            "INSTRUCTION",
            "REQUEST_LIMITS",
            "SCHEMA",
            "ImageError",
            "ParamError",
            "ParamSpec",
            "Params",
            "ParsedParams",
            "apply",
            "json_schema",
            "load_image",
            "parse_params",
            "save_image",
        ]
    )
    assert all(hasattr(executor, name) for name in executor.__all__)
