"""Lint the Python<->JS widget trait contract.

A Python trait the JS bridge (matterviz/extensions/anywidget/anywidget.ts)
doesn't read is silently dead (historic examples: ``missing_color``,
``ScatterPlotWidget.show_legend``); a bridge key without a Python trait is a
prop users cannot set. The bundle build writes its ``WIDGET_MODEL_KEYS`` contract
to ``widget-model-keys.json`` next to ``matterviz.js``; this test diffs it against
each widget's public synced traits in both directions. ``widget-model-schema.json``
gives each key's JSON kinds (``{"type": [...], "enum": [...]}``, typed from the
component props it drives), which each trait's traitlets type must fit.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError

import pytest
import traitlets as tl

import pymatviz.widgets
from pymatviz.widgets.matterviz import MatterVizWidget, fetch_widget_asset


# Widgets whose constructors reject empty input
_MINIMAL_KWARGS: dict[str, dict[str, Any]] = {
    "FermiSurfaceWidget": {"fermi_data": {"isosurfaces": []}}
}
CONTRACT_FILE = "widget-model-keys.json"
SCHEMA_FILE = "widget-model-schema.json"
WIDGET_CLASSES = sorted(
    (
        cls
        for cls in vars(pymatviz.widgets).values()
        if isinstance(cls, type)
        and issubclass(cls, MatterVizWidget)
        and cls is not MatterVizWidget
    ),
    key=lambda cls: cls.__name__,
)
JSON_KINDS = ("string", "number", "boolean", "object", "array")


def _load_bundle_json(filename: str) -> Any:
    """A bundle build file: a local build's when MATTERVIZ_ANYWIDGET_DIR is set
    (required there), else the pinned bundle's. Skips only when offline: a
    reachable host lacking the file fails the test.
    """
    if dev_dir := os.environ.get("MATTERVIZ_ANYWIDGET_DIR"):
        with open(f"{dev_dir}/{filename}", encoding="utf-8") as file:
            return json.load(file)
    try:
        return json.loads(fetch_widget_asset(filename))
    except FileNotFoundError as exc:
        cause = exc.__cause__
        if isinstance(cause, URLError) and not isinstance(cause, HTTPError):
            pytest.skip(f"offline, can't fetch {filename}: {cause}")
        raise


@pytest.fixture(scope="module")
def js_contract() -> dict[str, list[str]]:
    """The active bundle's {widget_type: model keys} contract."""
    return _load_bundle_json(CONTRACT_FILE)


@pytest.fixture(scope="module")
def js_schema() -> dict[str, dict[str, dict[str, list[str]]]]:
    """The active bundle's {widget_type: {model key: {type, enum?}}} schema."""
    return _load_bundle_json(SCHEMA_FILE)


def _trait_kinds(trait: tl.TraitType) -> tuple[set[str], list[str] | None]:
    """JSON kinds a trait accepts besides None, and its string enum values (None for
    an open string). tl.Any accepts every kind.
    """
    if isinstance(trait, tl.Union):
        kinds: set[str] = set()
        enums: list[str] = []
        open_string = False
        for member in trait.trait_types:
            member_kinds, member_enum = _trait_kinds(member)
            kinds |= member_kinds
            if member_enum is not None:
                enums += member_enum
            elif "string" in member_kinds:
                open_string = True
        return kinds, None if open_string or not enums else enums
    if isinstance(trait, tl.Enum):  # incl. CaselessStrEnum
        values = list(trait.values or [])
        kinds = {_value_kind(val) for val in values}
        str_values = [val for val in values if isinstance(val, str)]
        return kinds, str_values or None
    for trait_cls, kind in (
        (tl.Unicode, "string"),
        (tl.Bool, "boolean"),  # before Int: not an Int subclass, but explicit
        (tl.Float, "number"),
        (tl.Int, "number"),
        (tl.Dict, "object"),
        (tl.List, "array"),
        (tl.Tuple, "array"),
    ):
        if isinstance(trait, trait_cls):
            return {kind}, None
    if isinstance(trait, tl.Any) and type(trait) is tl.Any:
        return set(JSON_KINDS), None
    raise TypeError(f"no JSON kind for trait {trait!r}")


def _value_kind(value: object) -> str:
    """JSON kind of an enum value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    raise TypeError(f"no JSON kind for enum value {value!r}")


def _schema_mismatches(
    name: str, trait: tl.TraitType, schema: dict[str, list[str]]
) -> list[str]:
    """Ways a trait's type doesn't fit the JS schema descriptor of its key."""
    js_kinds = set(schema["type"])
    js_enum = schema.get("enum")
    py_kinds, py_enum = _trait_kinds(trait)
    problems: list[str] = []
    if type(trait) is tl.Any:
        if len(js_kinds - {"null"}) < 2:
            problems += [f"{name}: tl.Any where JS takes only {schema['type']}"]
    elif extra := py_kinds - js_kinds:
        problems += [f"{name}: Python allows {sorted(extra)}, JS takes {schema}"]
    # None reaches JS props as unset (the component default), so allow_none always
    # fits; a key whose prop takes null (e.g. color_bar=None hides it) needs it
    if "null" in js_kinds and not trait.allow_none:
        problems += [f"{name}: JS takes null but the trait rejects None"]
    if js_enum is not None and "string" in py_kinds and type(trait) is not tl.Any:
        if py_enum is None:
            problems += [f"{name}: open string, JS takes only {js_enum}"]
        elif sorted(py_enum) != sorted(js_enum):
            problems += [f"{name}: enum {sorted(py_enum)} != JS {js_enum}"]
    return problems


def test_trait_types_match_js_schema(
    js_contract: dict[str, list[str]],
    js_schema: dict[str, dict[str, dict[str, list[str]]]],
) -> None:
    """Each synced trait's traitlets type fits its key's JSON kinds in the bundle's
    schema: no kind JS can't take, None allowed where JS takes null, tl.Any only
    for keys of several kinds, and string enums equal to the JS literal union.
    """
    assert js_schema.keys() == js_contract.keys()
    mismatches: list[str] = []
    for cls in WIDGET_CLASSES:
        widget = cls(**_MINIMAL_KWARGS.get(cls.__name__, {}))
        key_schemas = js_schema.get(widget.widget_type, {})
        traits = widget.traits()
        for key in widget._synced_keys:
            if key in key_schemas:
                name = f"{cls.__name__}.{key}"
                mismatches += _schema_mismatches(name, traits[key], key_schemas[key])
    assert not mismatches, (
        "Python trait types don't fit the JS widget model schema (fix the trait in "
        "pymatviz/widgets or the prop type in matterviz):\n" + "\n".join(mismatches)
    )


def test_python_traits_match_js_contract(js_contract: dict[str, list[str]]) -> None:
    """Each widget's public synced traits exactly mirror the JS bridge's model keys,
    except base traits (widget_type, style, show_controls) a spec may ignore.
    """
    base_keys = set(MatterVizWidget.public_trait_names())
    drift: list[str] = []
    for cls in WIDGET_CLASSES:
        widget = cls(**_MINIMAL_KWARGS.get(cls.__name__, {}))
        if widget.widget_type not in js_contract:
            drift.append(f"{cls.__name__}: {widget.widget_type=} not in JS registry")
            continue
        py_keys = set(widget._synced_keys)
        js_keys = set(js_contract[widget.widget_type])
        if dead := py_keys - js_keys - base_keys:
            drift.append(
                f"{cls.__name__}: dead traits (JS never reads): {sorted(dead)}"
            )
        if missing := js_keys - py_keys:
            drift.append(
                f"{cls.__name__}: JS reads keys Python can't set: {sorted(missing)}"
            )
    assert not drift, (
        "Python<->JS widget trait contract drift (regenerate pymatviz/widgets/"
        "_schema_traits.py with `uv run tests/widgets/test_js_prop_parity.py`, or fix "
        "matterviz/extensions/anywidget/anywidget.ts or the widget class):\n"
        + "\n".join(drift)
    )


def test_public_trait_names_exclude_framework_traits() -> None:
    """public_trait_names() drops anywidget/ipywidgets traits but keeps a subclass's
    redeclaration of one (TrajectoryWidget.layout) and the base widget traits.
    """
    from anywidget import AnyWidget

    from pymatviz.widgets import StructureWidget, TrajectoryWidget

    framework = set(AnyWidget.class_traits(sync=True))
    for cls in (MatterVizWidget, StructureWidget):
        names = set(cls.public_trait_names())
        assert {"widget_type", "style", "show_controls"} <= names
        assert not names & framework
        assert not any(name.startswith("_") for name in names)
    assert "layout" in TrajectoryWidget.public_trait_names()
    widget = StructureWidget()
    assert widget._synced_keys == StructureWidget.public_trait_names()
    assert set(widget.to_dict()) == set(widget._synced_keys)


@pytest.mark.parametrize(
    ("trait", "schema", "n_problems"),
    [
        (tl.CaselessStrEnum(["a", "b"]), {"type": ["string"], "enum": ["a", "b"]}, 0),
        (tl.CaselessStrEnum(["a"]), {"type": ["string"], "enum": ["a", "b"]}, 1),
        (tl.Unicode(), {"type": ["string"], "enum": ["a"]}, 1),
        (tl.Bool(), {"type": ["string"], "enum": ["a"]}, 1),
        (tl.Int(allow_none=True), {"type": ["number"]}, 0),
        (tl.Dict(), {"type": ["object", "null"]}, 1),
        (tl.Dict(allow_none=True), {"type": ["object", "null"]}, 0),
        (tl.Any(), {"type": ["object"]}, 1),
        (tl.Any(), {"type": ["object", "array", "null"]}, 0),
        (tl.Union([tl.Bool(), tl.Dict()]), {"type": ["boolean", "object"]}, 0),
        (tl.Union([tl.Bool(), tl.List()]), {"type": ["boolean", "object"]}, 1),
    ],
)
def test_schema_mismatches(
    trait: tl.TraitType, schema: dict[str, list[str]], n_problems: int
) -> None:
    """_schema_mismatches flags extra kinds, None where JS needs null, lone-kind
    tl.Any and string enums that differ from (or are looser than) the JS union.
    """
    assert len(_schema_mismatches("key", trait, schema)) == n_problems


_KIND_TRAITS = {"boolean": "tl.Bool", "number": "tl.Float", "string": "tl.Unicode"}
_KIND_TRAITS |= {"object": "tl.Dict", "array": "tl.List"}


def _trait_source(spec: dict[str, list[str]]) -> str:
    """Source of a None-default synced trait taking a model key's JSON kinds."""
    members = [
        f"tl.CaselessStrEnum({spec['enum']})"
        if kind == "string" and "enum" in spec
        else f"{trait}()"
        for kind, trait in _KIND_TRAITS.items()
        if kind in spec["type"]
    ]
    if len(members) > 1:
        return f"optional_trait(tl.Union, trait_types=[{', '.join(members)}])"
    if members[0].startswith("tl.CaselessStrEnum"):
        return f"optional_trait(tl.CaselessStrEnum, values={spec['enum']})"
    return f"optional_trait({members[0].removesuffix('()')})"


def schema_traits_source(schema: dict[str, dict[str, dict[str, list[str]]]]) -> str:
    """Source of pymatviz/widgets/_schema_traits.py: a traits mixin per widget type
    with a trait per model key, except MatterVizWidget's own (style, show_controls).
    """
    base_keys = set(MatterVizWidget.public_trait_names())
    lines = [
        '"""Synced traits of each MatterViz widget\'s model keys, generated from the',
        "bundle's widget-model-schema.json by",
        "`uv run tests/widgets/test_js_prop_parity.py`. Don't edit (excluded from",
        "ruff): widget classes override traits that need defaults or validation.",
        '"""',
        "",
        "import traitlets as tl",
        "",
        "from pymatviz.widgets._traits import optional_trait",
    ]
    for widget_type, keys in schema.items():
        cls_name = "".join(map(str.capitalize, widget_type.split("_"))) + "Traits"
        lines += ["", "", f"class {cls_name}(tl.HasTraits):"]
        lines += [f'    """{widget_type} widget model keys."""', ""]
        lines += [f'    widget_type = tl.Unicode("{widget_type}").tag(sync=True)']
        lines += [
            f"    {key} = {_trait_source(spec)}"
            for key, spec in keys.items()
            if key not in base_keys
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out_path = f"{os.path.dirname(pymatviz.widgets.__file__)}/_schema_traits.py"
    # build first: a failed fetch must not leave an empty, unimportable module
    source = schema_traits_source(_load_bundle_json(SCHEMA_FILE))
    with open(out_path, "w", encoding="utf-8") as file:
        file.write(source)
    print(f"wrote {out_path}")  # noqa: T201
