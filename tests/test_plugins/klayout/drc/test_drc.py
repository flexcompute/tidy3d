"""Tests tidy3d/plugins/klayout/drc/drc.py"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from pydantic import ValidationError

import tidy3d as td
from tidy3d.exceptions import FileError
from tidy3d.plugins.klayout.drc.drc import DRCConfig, DRCRunner, run_drc_on_gds
from tidy3d.plugins.klayout.drc.results import (
    DRCResults,
    DRCViolation,
    EdgeMarker,
    EdgePairMarker,
    MultiPolygonMarker,
    PolygonMarker,
    parse_violation_value,
)

filepath = Path(os.path.dirname(os.path.abspath(__file__)))
KLAYOUT_PLUGIN_PATH = "tidy3d.plugins.klayout"


def _basic_drc_config_kwargs(tmp_path: Path) -> dict[str, Path | bool]:
    """Return minimal kwargs needed to instantiate DRCConfig in tests."""

    drc_runset = tmp_path / "test.drc"
    drc_runset.write_text('source($gdsfile)\nreport("DRC", $resultsfile)\n')
    gdsfile = tmp_path / "test.gds"
    gdsfile.write_text("")
    resultsfile = tmp_path / "results.lyrdb"
    return {
        "gdsfile": gdsfile,
        "drc_runset": drc_runset,
        "resultsfile": resultsfile,
        "verbose": False,
    }


def _write_results_file(
    tmp_path: Path,
    *,
    category: str = "min_width",
    num_items: int = 1,
    filename: str = "many_results.lyrdb",
    cells: tuple[str, ...] | None = None,
) -> Path:
    """Write a simple DRC results file with the requested number of items."""

    template_header = f"""\
<?xml version=\"1.0\" encoding=\"utf-8\"?>
<report-database>
 <categories>
  <category>
   <name>{category}</name>
   <description>auto</description>
   <categories></categories>
  </category>
 </categories>
 <cells></cells>
 <items>
"""
    template_footer = """\
 </items>
</report-database>
"""
    item = """\
  <item>
   <tags/>
   <category>{category}</category>
   <cell>{cell}</cell>
   <visited>false</visited>
   <multiplicity>1</multiplicity>
   <comment/>
   <image/>
   <values>
    <value>edge: (0.0,0.0;1.0,1.0)</value>
   </values>
  </item>
"""
    contents = [template_header]
    for idx in range(num_items):
        cell = "TOP"
        if cells is not None and idx < len(cells):
            cell = cells[idx]
        contents.append(item.format(category=category, cell=cell))
    contents.append(template_footer)
    path = tmp_path / filename
    path.write_text("".join(contents))
    return path


def _capture_log_warnings(monkeypatch):
    """Capture calls to td.log.warning."""

    messages = []

    def fake_warning(message, *args, **kwargs):
        messages.append(message % args if args else message)

    monkeypatch.setattr(td.log, "warning", fake_warning)
    return messages


def test_runner_passes_drc_args_to_config(monkeypatch, tmp_path):
    """Ensure DRCRunner forwards drc_args into the generated DRCConfig."""

    drc_runset = tmp_path / "test.drc"
    drc_runset.write_text('source($gdsfile)\nreport("DRC", $resultsfile)\n')
    gdsfile = tmp_path / "test.gds"
    gdsfile.write_text("")
    resultsfile = tmp_path / "results.lyrdb"
    captured_config = {}

    def mock_run_drc_on_gds(config, **_kwargs):
        captured_config["config"] = config
        return DRCResults.load(filepath / "fixtures" / "drc_results.lyrdb")

    monkeypatch.setattr(f"{KLAYOUT_PLUGIN_PATH}.drc.drc.run_drc_on_gds", mock_run_drc_on_gds)

    runner = DRCRunner(drc_runset=drc_runset, verbose=False)
    user_args = {"foo": "bar", "baz": "1"}
    runner.run(source=gdsfile, resultsfile=resultsfile, drc_args=user_args)

    assert captured_config["config"].drc_args == user_args


def test_run_drc_on_gds_appends_custom_args(monkeypatch, tmp_path):
    """run_drc_on_gds adds extra -rd pairs for drc_args."""

    drc_runset = tmp_path / "test.drc"
    drc_runset.write_text('source($gdsfile)\nreport("DRC", $resultsfile)\n')
    gdsfile = tmp_path / "test.gds"
    gdsfile.write_text("")
    resultsfile = tmp_path / "results.lyrdb"

    monkeypatch.setattr(f"{KLAYOUT_PLUGIN_PATH}.drc.drc.check_installation", lambda **_: None)

    captured_cmd = {}

    class DummyCompleted:
        def __init__(self):
            self.returncode = 0
            self.stdout = b""
            self.stderr = b""

    def fake_run(cmd, capture_output):
        captured_cmd["cmd"] = cmd
        return DummyCompleted()

    monkeypatch.setattr(f"{KLAYOUT_PLUGIN_PATH}.drc.drc.run", fake_run)
    monkeypatch.setattr(
        f"{KLAYOUT_PLUGIN_PATH}.drc.drc.DRCResults.load",
        lambda resultsfile, **_kwargs: DRCResults(violations_by_category={}),
    )

    config = DRCConfig(
        gdsfile=gdsfile,
        drc_runset=drc_runset,
        resultsfile=resultsfile,
        verbose=False,
        drc_args={"string_arg": "text", "numeric_value": 1},
    )

    run_drc_on_gds(config)

    expected_tail = ["-rd", "string_arg=text", "-rd", "numeric_value=1"]
    assert captured_cmd["cmd"][-len(expected_tail) :] == expected_tail


def test_drc_config_args_require_mapping(tmp_path):
    """drc_args must be a mapping and refuses other iterables."""

    kwargs = _basic_drc_config_kwargs(tmp_path)
    with pytest.raises(ValidationError):
        DRCConfig(**kwargs, drc_args=["not", "a", "mapping"])


def test_drc_config_args_reject_reserved_keys(tmp_path):
    """Reserved keys such as gdsfile cannot be overridden via drc_args."""

    kwargs = _basic_drc_config_kwargs(tmp_path)
    with pytest.raises(ValidationError):
        DRCConfig(**kwargs, drc_args={"gdsfile": "custom.gds"})


def test_drc_config_args_stringify_values(tmp_path):
    """Non-string keys and values are coerced to strings by the validator."""

    kwargs = _basic_drc_config_kwargs(tmp_path)
    config = DRCConfig(**kwargs, drc_args={1: Path("foo"), "flag": True})

    assert config.drc_args == {"1": "foo", "flag": "True"}


def test_drc_config_args_unstringifiable_value(tmp_path):
    """Non-stringifiable drc_args values should raise a ValidationError."""

    class Unstringifiable:
        def __str__(self):
            raise RuntimeError("cannot stringify")

    kwargs = _basic_drc_config_kwargs(tmp_path)

    with pytest.raises(
        ValidationError, match="Could not coerce keys and values of drc_args to strings."
    ):
        DRCConfig(**kwargs, drc_args={"bad": Unstringifiable()})


class TestDRCRunner:
    """Test DRCRunner"""

    @staticmethod
    def write_drcrunset(tmp_path, drcrunset_name, drcrunset_content):
        """Write a DRC file to a temporary path"""
        with Path(tmp_path / drcrunset_name).open("w") as f:
            f.write(drcrunset_content)

    @staticmethod
    @pytest.fixture(scope="class")
    def good_drcrunset_content():
        """The content of a valid DRC file"""
        return """
        source($gdsfile)
        report("DRC results", $resultsfile)
        """

    @staticmethod
    def wrap_drc_to_lydrc(body: str):
        """Return the XML-wrapped .lydrc runset content."""
        xml = f"""\
        <?xml version="1.0" encoding="utf-8"?>
        <klayout-macro>
        <description>Test DRC runset</description>
        <version/>
        <category>drc</category>
        <prolog/>
        <epilog/>
        <text>
        {body}
        </text>
        </klayout-macro>
        """

        return xml

    @staticmethod
    @pytest.fixture(scope="class")
    def bad_drcrunset_content_source():
        """The content of a DRC file with a bad source declaration"""
        return """
        source($gfdsfile)
        report("DRC results", $resultsfile)
        """

    @staticmethod
    @pytest.fixture(scope="class")
    def bad_drcrunset_content_report():
        """The content of a DRC file with a bad report declaration"""
        return """
        source($gdsfile)
        report("DRC results", $refsultsfile)
        """

    @staticmethod
    def write_gdsfile(tmp_path, gdsfile_name):
        """Write a geometry to a GDS file"""
        geom = TestDRCRunner.make_geom()
        geom.to_gds_file(tmp_path / gdsfile_name, **TestDRCRunner.geom_to_gds_kwargs())

    @staticmethod
    @pytest.fixture(scope="class")
    def geom():
        """Make a simple geometry"""
        vertices = [(-2, 0), (-1, 1), (0, 0.5), (1, 1), (2, 0), (0, -1)]
        return td.PolySlab(vertices=vertices, slab_bounds=(0, 0.22), axis=2)

    @staticmethod
    @pytest.fixture(scope="class")
    def geom_to_gds_kwargs():
        """The kwargs to pass to the geometry's to_gds_file() method"""
        return {"z": 0.1, "gds_layer": 0, "gds_dtype": 0}

    @staticmethod
    @pytest.fixture(scope="class")
    def structure(geom):
        """Make a structure with a geometry"""
        return td.Structure(geometry=geom, medium=td.Medium(permittivity=12))

    @staticmethod
    @pytest.fixture(scope="class")
    def structure_to_gds_kwargs():
        """The kwargs to pass to the structure's to_gds_file() method"""
        return {"z": 0.1, "gds_layer": 0, "gds_dtype": 0}

    @staticmethod
    @pytest.fixture(scope="class")
    def sim(structure):
        """Make a simulation with a structure"""
        return td.Simulation(
            size=(10, 10, 1),
            grid_spec=td.GridSpec.uniform(dl=0.02),
            structures=[structure],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
            run_time=1e-12,
        )

    @staticmethod
    @pytest.fixture(scope="class")
    def sim_to_gds_kwargs():
        """The kwargs to pass to the simulation's to_gds_file() method"""
        return {"z": 0.1, "gds_layer_dtype_map": {td.Medium(permittivity=12): (0, 0)}}

    @staticmethod
    def run(
        monkeypatch,
        drc_runsetfile,
        verbose,
        source,
        td_object_gds_savefile,
        resultsfile,
        drc_args=None,
        **to_gds_file_kwargs,
    ):
        """Calls DRCRunner.run with dummy run_drc_on_gds()"""

        # monkeypatch run_drc_on_gds() since the test machines do not have KLayout installed
        def mock_run_drc_on_gds(config, **_kwargs):
            return DRCResults.load(filepath / "fixtures" / "drc_results.lyrdb")

        monkeypatch.setattr(f"{KLAYOUT_PLUGIN_PATH}.drc.drc.run_drc_on_gds", mock_run_drc_on_gds)

        runner = DRCRunner(drc_runset=drc_runsetfile, verbose=verbose)
        return runner.run(
            source=source,
            td_object_gds_savefile=td_object_gds_savefile,
            resultsfile=resultsfile,
            drc_args=drc_args,
            **to_gds_file_kwargs,
        )

    @pytest.mark.parametrize("verbose", [True, False])
    @pytest.mark.parametrize("drc_file_suffix", [".drc", ".lydrc"])
    def test_valid_run_on_gds(
        self,
        monkeypatch,
        tmp_path,
        verbose,
        geom,
        geom_to_gds_kwargs,
        good_drcrunset_content,
        drc_file_suffix,
    ):
        """Test that no error is raised when runs on a gds are valid"""
        geom.to_gds_file(tmp_path / "test.gds", **geom_to_gds_kwargs)
        drc_content = good_drcrunset_content
        if drc_file_suffix == ".lydrc":
            drc_content = TestDRCRunner.wrap_drc_to_lydrc(drc_content)
        self.write_drcrunset(tmp_path, f"good_drcfile{drc_file_suffix}", drc_content)
        self.run(
            monkeypatch=monkeypatch,
            drc_runsetfile=tmp_path / f"good_drcfile{drc_file_suffix}",
            verbose=verbose,
            source=tmp_path / "test.gds",
            td_object_gds_savefile=tmp_path / "test.gds",
            resultsfile=filepath / "fixtures" / "drc_results.lyrdb",
        )

    @pytest.mark.parametrize("verbose", [True, False])
    @pytest.mark.parametrize("drc_file_suffix", [".drc", ".lydrc"])
    @pytest.mark.parametrize(
        "td_object, obj_to_gds_kwargs",
        [
            ("geom", "geom_to_gds_kwargs"),
            ("structure", "structure_to_gds_kwargs"),
            ("sim", "sim_to_gds_kwargs"),
        ],
    )
    def test_valid_run_on_td_object(
        self,
        request,
        monkeypatch,
        tmp_path,
        verbose,
        td_object,
        obj_to_gds_kwargs,
        good_drcrunset_content,
        drc_file_suffix,
    ):
        """Test that no error is raised when runs on a Geometry, Structure, or Simulation are valid"""
        drc_content = good_drcrunset_content
        if drc_file_suffix == ".lydrc":
            drc_content = TestDRCRunner.wrap_drc_to_lydrc(drc_content)
        self.write_drcrunset(tmp_path, f"good_drcfile{drc_file_suffix}", drc_content)
        self.run(
            monkeypatch=monkeypatch,
            drc_runsetfile=tmp_path / f"good_drcfile{drc_file_suffix}",
            verbose=verbose,
            source=request.getfixturevalue(td_object),
            td_object_gds_savefile=tmp_path / "test.gds",
            resultsfile=filepath / "fixtures" / "drc_results.lyrdb",
            **request.getfixturevalue(obj_to_gds_kwargs),
        )

    @pytest.mark.parametrize(
        "bad_drcrunset_content", ["bad_drcrunset_content_source", "bad_drcrunset_content_report"]
    )
    @pytest.mark.parametrize("drc_file_suffix", [".drc", ".lydrc"])
    def test_check_drcfile_format_invalid(
        self,
        request,
        monkeypatch,
        tmp_path,
        geom,
        geom_to_gds_kwargs,
        bad_drcrunset_content,
        drc_file_suffix,
    ):
        """Tests that ValidationError is raised when the drc file content is invalid"""
        geom.to_gds_file(tmp_path / "test.gds", **geom_to_gds_kwargs)
        drc_content = request.getfixturevalue(bad_drcrunset_content)
        if drc_file_suffix == ".lydrc":
            drc_content = TestDRCRunner.wrap_drc_to_lydrc(drc_content)
        self.write_drcrunset(tmp_path, f"bad_drcrunset{drc_file_suffix}", drc_content)
        with pytest.raises(ValidationError) as e:
            self.run(
                monkeypatch=monkeypatch,
                drc_runsetfile=tmp_path / f"bad_drcrunset{drc_file_suffix}",
                verbose=True,
                source=tmp_path / "test.gds",
                td_object_gds_savefile=None,
                resultsfile=filepath / "fixtures" / "drc_results.lyrdb",
            )

    def test_check_gdsfile_exists(self, monkeypatch, tmp_path, good_drcrunset_content):
        """Test gdsfile existence checking works"""
        self.write_drcrunset(tmp_path, "good_drcfile.drc", good_drcrunset_content)
        with pytest.raises(ValidationError):
            self.run(
                monkeypatch=monkeypatch,
                drc_runsetfile=tmp_path / "good_drcfile.drc",
                verbose=True,
                source=tmp_path / "test.gds",
                td_object_gds_savefile=None,
                resultsfile=filepath / "fixtures" / "drc_results.lyrdb",
            )

    def test_check_gdsfile_filetype(
        self, monkeypatch, tmp_path, good_drcrunset_content, geom, geom_to_gds_kwargs
    ):
        """Test gdsfile filetype checking works"""
        self.write_drcrunset(tmp_path, "good_drcfile.drc", good_drcrunset_content)
        geom.to_gds_file(tmp_path / "test.g2ds", **geom_to_gds_kwargs)
        with pytest.raises(ValidationError):
            self.run(
                monkeypatch=monkeypatch,
                drc_runsetfile=tmp_path / "good_drcfile.drc",
                verbose=True,
                source=tmp_path / "test.g2ds",
                td_object_gds_savefile=None,
                resultsfile=filepath / "fixtures" / "drc_results.lyrdb",
            )

    def test_check_designrulefile_exists(self, monkeypatch, tmp_path, geom, geom_to_gds_kwargs):
        """Test design rule file existence checking works"""
        geom.to_gds_file(tmp_path / "test.gds", **geom_to_gds_kwargs)
        with pytest.raises(ValidationError):
            self.run(
                monkeypatch=monkeypatch,
                drc_runsetfile=tmp_path / "not_a_drc_file.drc",
                verbose=True,
                source=tmp_path / "test.gds",
                td_object_gds_savefile=None,
                resultsfile=filepath / "fixtures" / "drc_results.lyrdb",
            )

    def test_check_designrulefile_filetype(
        self, monkeypatch, tmp_path, geom, geom_to_gds_kwargs, good_drcrunset_content
    ):
        """Test design rule file filetype checking works"""
        geom.to_gds_file(tmp_path / "test.gds", **geom_to_gds_kwargs)
        self.write_drcrunset(tmp_path, "good_drcfile.drc2", good_drcrunset_content)
        with pytest.raises(ValidationError):
            self.run(
                monkeypatch=monkeypatch,
                drc_runsetfile=tmp_path / "good_drcfile.drc2",
                verbose=True,
                source=tmp_path / "test.gds",
                td_object_gds_savefile=None,
                resultsfile=filepath / "fixtures" / "drc_results.lyrdb",
            )


class TestDRCResults:
    """Test DRCResults"""

    @pytest.fixture(scope="class")
    def drc_results(self):
        """Load the DRC results"""
        return DRCResults.load(filepath / "fixtures" / "drc_results.lyrdb")

    @pytest.fixture(scope="class")
    def drc_results_clean(self):
        """Load the DRC results"""
        return DRCResults.load(filepath / "fixtures" / "drc_results_clean.lyrdb")

    def test_result_file_load(self, tmp_path):
        """Test that result file loading works"""
        # this should not raise an error
        DRCResults.load(filepath / "fixtures" / "drc_results.lyrdb")

        # file not found
        with pytest.raises(FileError):
            DRCResults.load(tmp_path / "not_a_results_file.lyrdb")

        # not xml file
        with Path(tmp_path / "bad_resultsfile.lyrdb").open("w") as f:
            f.write("""
        not a valid xml file
        """)
        with pytest.raises(ET.ParseError):
            DRCResults.load(tmp_path / "bad_resultsfile.lyrdb")

    def test_is_drc_clean(self, drc_results, drc_results_clean):
        """Test DRCResults.is_clean"""
        assert not drc_results.is_clean
        assert drc_results_clean.is_clean

    def test_count_drc_violations(self, drc_results):
        """Test that counting violations works"""
        assert drc_results["min_width"].count == 2
        assert drc_results["min_gap"].count == 2
        assert drc_results["min_area"].count == 1
        assert drc_results["min_hole"].count == 1

    def test_drc_result_markers(self, drc_results):
        """Test that the DRC result markers are correct"""
        assert drc_results["min_width"].markers[0].edge_pair[0] == ((-0.6, 0.163), (-0.6, 0.419))
        assert drc_results["min_width"].markers[0].edge_pair[1] == ((-0.31, 0.342), (-0.31, 0.24))
        assert drc_results["min_width"].markers[1].edge_pair[0] == ((-0.206, 0.342), (-0.31, 0.342))
        assert drc_results["min_width"].markers[1].edge_pair[1] == ((-0.521, 0.555), (0.005, 0.555))
        assert drc_results["min_gap"].markers[0].edge_pair[0] == ((-0.31, 0.24), (-0.206, 0.24))
        assert drc_results["min_gap"].markers[0].edge_pair[1] == ((-0.206, 0.342), (-0.31, 0.342))
        assert drc_results["min_gap"].markers[1].edge_pair[0] == ((-0.206, 0.24), (-0.206, 0.342))
        assert drc_results["min_gap"].markers[1].edge_pair[1] == ((-0.31, 0.342), (-0.31, 0.24))
        assert drc_results["min_area"].markers[0].hull == (
            (-0.6, -0.112),
            (-0.6, 0.555),
            (0.217, 0.555),
            (0.217, -0.112),
        )
        assert len(drc_results["min_area"].markers[0].holes) == 1
        assert drc_results["min_area"].markers[0].holes[0] == (
            (-0.31, 0.24),
            (-0.206, 0.24),
            (-0.206, 0.342),
            (-0.31, 0.342),
        )
        assert drc_results["min_hole"].markers[0].hull == (
            (-0.31, 0.24),
            (-0.31, 0.342),
            (-0.206, 0.342),
            (-0.206, 0.24),
        )
        assert len(drc_results["min_hole"].markers[0].holes) == 0
        for violation in drc_results.violations_by_category.values():
            for marker in violation.markers:
                assert marker.cell == "TOP"

    def test_drc_violation_cell_helpers(self):
        """DRCViolation provides cell-aware helpers."""
        violation = DRCViolation(
            category="cat_a",
            markers=(
                EdgeMarker(cell="CELL_A", edge=((0.0, 0.0), (1.0, 1.0))),
                EdgeMarker(cell="CELL_B", edge=((1.0, 1.0), (2.0, 2.0))),
                EdgeMarker(cell="CELL_A", edge=((0.5, 0.5), (1.5, 1.5))),
            ),
        )
        assert violation.violated_cells == ("CELL_A", "CELL_B")

        by_cell = violation.violations_by_cell
        assert set(by_cell) == {"CELL_A", "CELL_B"}
        assert by_cell["CELL_B"].count == 1

        markers_cell_a = by_cell["CELL_A"].markers
        assert all(marker.cell == "CELL_A" for marker in markers_cell_a)

    def test_drc_results_cell_helpers(self):
        """DRCResults aggregates violations across cells."""
        violation_a = DRCViolation(
            category="cat_a",
            markers=(
                EdgeMarker(cell="CELL_A", edge=((0.0, 0.0), (1.0, 1.0))),
                EdgeMarker(cell="CELL_B", edge=((1.0, 1.0), (2.0, 2.0))),
            ),
        )
        violation_b = DRCViolation(
            category="cat_b",
            markers=(
                EdgeMarker(cell="CELL_B", edge=((2.0, 2.0), (3.0, 3.0))),
                EdgeMarker(cell="CELL_C", edge=((3.0, 3.0), (4.0, 4.0))),
            ),
        )
        results = DRCResults(
            violations_by_category={
                "cat_a": violation_a,
                "cat_b": violation_b,
            }
        )
        assert results.violated_cells == ("CELL_A", "CELL_B", "CELL_C")

        violations_by_cell = results.violations_by_cell
        assert set(violations_by_cell) == {"CELL_A", "CELL_B", "CELL_C"}
        assert len(violations_by_cell["CELL_A"]) == 1
        assert len(violations_by_cell["CELL_C"]) == 1

        cell_b_violations = violations_by_cell["CELL_B"]
        assert {violation.category for violation in cell_b_violations} == {"cat_a", "cat_b"}
        for violation in cell_b_violations:
            assert all(marker.cell == "CELL_B" for marker in violation.markers)

    @pytest.mark.parametrize(
        "edge_value, expected_edge",
        [
            ("edge: (1.0,2.0;3.0,4.0)", ((1.0, 2.0), (3.0, 4.0))),
            ("edge: (-1.0,-2.0;-3.0,-4.0)", ((-1.0, -2.0), (-3.0, -4.0))),
        ],
    )
    def test_parse_edge(self, edge_value, expected_edge):
        """Test parsing edge violation values."""
        edge_result = parse_violation_value(edge_value, cell="TEST_CELL")
        assert edge_result.edge == expected_edge
        assert edge_result.cell == "TEST_CELL"

    def test_parse_edge_pair(self):
        """Test parsing edge-pair violation values."""
        edge_pair_value = "edge-pair: (1.0,2.0;3.0,4.0)|(5.0,6.0;7.0,8.0)"
        edge_pair_result = parse_violation_value(edge_pair_value, cell="TEST_CELL")
        assert edge_pair_result.edge_pair[0] == ((1.0, 2.0), (3.0, 4.0))
        assert edge_pair_result.edge_pair[1] == ((5.0, 6.0), (7.0, 8.0))
        assert edge_pair_result.cell == "TEST_CELL"
        assert edge_pair_result.symmetric is True

    @pytest.mark.parametrize(
        "value, expected_symmetric",
        [
            ("edge-pair: (1.0,2.0;3.0,4.0)|(5.0,6.0;7.0,8.0)", True),
            ("edge-pair: (1.0,2.0;3.0,4.0)/(5.0,6.0;7.0,8.0)", False),
        ],
    )
    def test_parse_edge_pair_symmetric_flag(self, value, expected_symmetric):
        """Test that '|' yields symmetric=True and '/' yields symmetric=False."""
        result = parse_violation_value(value, cell="TEST_CELL")
        assert isinstance(result, EdgePairMarker)
        assert result.symmetric is expected_symmetric
        assert result.edge_pair[0] == ((1.0, 2.0), (3.0, 4.0))
        assert result.edge_pair[1] == ((5.0, 6.0), (7.0, 8.0))

    def test_parse_edge_pair_directed_negative_coords(self):
        """Test parsing directed edge-pair with negative coordinates."""
        value = "edge-pair: (-1.5,-2.0;-3.0,-4.0)/(-5.0,-6.0;-7.0,-8.0)"
        result = parse_violation_value(value, cell="TEST_CELL")
        assert isinstance(result, EdgePairMarker)
        assert result.symmetric is False
        assert result.edge_pair[0] == ((-1.5, -2.0), (-3.0, -4.0))
        assert result.edge_pair[1] == ((-5.0, -6.0), (-7.0, -8.0))

    def test_parse_polygon(self):
        """Test parsing a single polygon violation string."""
        polygon_value = "polygon: (1.0,2.0;3.0,4.0;5.0,6.0;1.0,2.0)"
        polygon_result = parse_violation_value(polygon_value, cell="TEST_CELL")
        assert isinstance(polygon_result, PolygonMarker)
        assert polygon_result.hull == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (1.0, 2.0))
        assert polygon_result.holes == ()
        assert polygon_result.cell == "TEST_CELL"

    def test_parse_polygon_with_hole(self):
        """Test parsing a polygon with one hole (hull + hole)."""
        polygon_value = (
            "polygon: (1.0,2.0;3.0,4.0;5.0,6.0;1.0,2.0/7.0,8.0;9.0,10.0;11.0,12.0;7.0,8.0)"
        )
        polygon_result = parse_violation_value(polygon_value, cell="TEST_CELL")
        assert isinstance(polygon_result, PolygonMarker)
        assert polygon_result.hull == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (1.0, 2.0))
        assert len(polygon_result.holes) == 1
        assert polygon_result.holes[0] == ((7.0, 8.0), (9.0, 10.0), (11.0, 12.0), (7.0, 8.0))
        assert polygon_result.cell == "TEST_CELL"

    def test_parse_polygon_with_multiple_holes(self):
        """Test parsing a polygon with multiple holes."""
        polygon_value = "polygon: (0,0;10,0;10,10;0,10/2,2;4,2;4,4;2,4/6,6;8,6;8,8;6,8)"
        polygon_result = parse_violation_value(polygon_value, cell="TEST_CELL")
        assert isinstance(polygon_result, PolygonMarker)
        assert polygon_result.hull == ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
        assert len(polygon_result.holes) == 2
        assert polygon_result.holes[0] == ((2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0))
        assert polygon_result.holes[1] == ((6.0, 6.0), (8.0, 6.0), (8.0, 8.0), (6.0, 8.0))

    def test_multi_polygon_marker_deprecation_alias(self):
        """MultiPolygonMarker still works as a subclass of PolygonMarker."""
        marker = MultiPolygonMarker(
            cell="TEST_CELL",
            hull=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            holes=(((0.2, 0.2), (0.4, 0.2), (0.4, 0.4), (0.2, 0.4)),),
        )
        assert isinstance(marker, PolygonMarker)
        assert isinstance(marker, MultiPolygonMarker)
        # .polygons backwards-compat property returns (hull,) + holes
        assert len(marker.polygons) == 2
        assert marker.polygons[0] == marker.hull
        assert marker.polygons[1] == marker.holes[0]

    @pytest.mark.parametrize(
        "invalid_edge",
        [
            "edge: invalid_format",
            "edge: (1.,3.,4.,1.)",
            "edge: (1.0,2.0;3.0,4.0;5.0,6.0)",
            "edge: (1.0,;3.0,)",
            "edge: (1.0,2.0;3e.0,4.0)",
            "edge: (1.0,;3.0,4.0)",
        ],
    )
    def test_parse_invalid_edge_format(self, invalid_edge):
        """Test parsing invalid violation format."""
        with pytest.raises(ValueError):
            parse_violation_value(invalid_edge, cell="TEST_CELL")

    @pytest.mark.parametrize(
        "invalid_edge_pair",
        [
            "edge-pair: (1.0,2.0;3.0,4.0)|(5.0,6.0;7.0,8.0;9.0,10.0)",
            "edge-pair: (1b.0,2.0;3.0,4.0)|(5.0,6.0;7.0,8.0)",
            "edge-pair: (1.0,2.0;3.0,4.0)|(5.0,)",
        ],
    )
    def test_parse_invalid_edge_pair_format(self, invalid_edge_pair):
        """Test parsing invalid edge-pair violation format."""
        with pytest.raises(ValueError):
            parse_violation_value(invalid_edge_pair, cell="TEST_CELL")

    @pytest.mark.parametrize(
        "invalid_polygon",
        [
            "polygon: (1b.0,2.0;3.0,4.0;5.0,6.0;1.0,2.0;1.0,2.0)",
            "polygon: (1.0,2.0;3.0,4.0;|5.0,6.0;1.0,2.0;1.0,2.0)",
            "polygon: (1.0,;3.0,4.0;5.0,6.0;1.0,2.0;1.0,2.0)",
        ],
    )
    def test_parse_invalid_polygon_format(self, invalid_polygon):
        """Test parsing invalid polygon violation format."""
        with pytest.raises(ValueError):
            parse_violation_value(invalid_polygon, cell="TEST_CELL")

    @pytest.mark.parametrize(
        "invalid_polygons",
        [
            "polygon: (1.0,2.0;3.0,4.0;5.0,6.0;1.0,2.0//7.0,8.0;9.0,10.0;11.0,12.0;7.0,8.0)",
            "polygon: (1.0,2.0;3.0,4.0;5.0,6.0;1.0,2.0/",
            "polygon: (1.0,2.0;3.0,4/.0;5.0,6.0;1.0,2.0)",
        ],
    )
    def test_parse_invalid_polygon_format_multiple_polygons(self, invalid_polygons):
        """Test parsing invalid polygon violation format with multiple polygons."""
        with pytest.raises(ValueError) as e:
            parse_violation_value(invalid_polygons, cell="TEST_CELL")

    def test_parse_violation_value_unknown_type(self):
        """Test parsing unknown violation type."""
        with pytest.raises(ValueError):
            parse_violation_value("unknown: (1.0,2.0)", cell="TEST_CELL")

    def test_results_warn_without_limit(self, monkeypatch, tmp_path):
        """Warn when no limit is set and a category exceeds the threshold."""

        warnings = _capture_log_warnings(monkeypatch)
        monkeypatch.setattr(
            f"{KLAYOUT_PLUGIN_PATH}.drc.results.UNLIMITED_VIOLATION_WARNING_COUNT",
            3,
        )
        results_path = _write_results_file(tmp_path, num_items=4)
        results = DRCResults.load(results_path)
        assert results["min_width"].count == 4
        assert len(warnings) == 1
        assert "many markers (4)" in warnings[0]

    def test_results_warn_when_limit_truncates(self, monkeypatch, tmp_path):
        """Warn when the global limit removes markers."""

        warnings = _capture_log_warnings(monkeypatch)
        results_path = _write_results_file(tmp_path, category="overflow", num_items=4)
        results = DRCResults.load(results_path, max_results=2)
        assert results["overflow"].count == 2
        assert len(warnings) == 1
        assert "only the first 2" in warnings[0]

    def test_siepic_fixture_integration(self):
        """Full integration test for the SiEPIC fixture covering all marker types and DRCResults API."""
        results = DRCResults.load(filepath / "fixtures" / "siepic_ebeam_violations.lyrdb")

        # --- DRCResults-level assertions ---
        assert not results.is_clean
        assert results.categories == (
            "Si_width",
            "Si_space",
            "M1_width",
            "M1_space",
            "M2_width",
            "M2_space",
            "M2_M1_overlap",
            "DT_Metal_separation",
            "Si_boundary",
        )
        assert results.violation_counts == {
            "Si_width": 3,
            "Si_space": 2,
            "M1_width": 0,
            "M1_space": 0,
            "M2_width": 0,
            "M2_space": 0,
            "M2_M1_overlap": 1,
            "DT_Metal_separation": 4,
            "Si_boundary": 1,
        }
        assert results.violated_cells == ("TOP",)

        # --- Symmetric edge-pairs from same-layer rules ---
        si_width = results["Si_width"]
        assert si_width.count == 3
        for marker in si_width.markers:
            assert isinstance(marker, EdgePairMarker)
            assert marker.symmetric is True

        si_space = results["Si_space"]
        assert si_space.count == 2
        for marker in si_space.markers:
            assert isinstance(marker, EdgePairMarker)
            assert marker.symmetric is True

        # --- Directed edge-pairs from cross-layer rules ---
        m2_m1_overlap = results["M2_M1_overlap"]
        assert m2_m1_overlap.count == 1
        assert isinstance(m2_m1_overlap.markers[0], EdgePairMarker)
        assert m2_m1_overlap.markers[0].symmetric is False

        dt_metal_sep = results["DT_Metal_separation"]
        assert dt_metal_sep.count == 4
        for marker in dt_metal_sep.markers:
            assert isinstance(marker, EdgePairMarker)
            assert marker.symmetric is False

        # --- Polygon from boundary check ---
        si_boundary = results["Si_boundary"]
        assert si_boundary.count == 1
        assert isinstance(si_boundary.markers[0], PolygonMarker)
        assert si_boundary.markers[0].hull == (
            (205.0, 0.0),
            (205.0, 5.0),
            (210.0, 5.0),
            (210.0, 0.0),
        )
        assert si_boundary.markers[0].holes == ()

        # --- Clean categories have zero violations ---
        for cat in ("M1_width", "M1_space", "M2_width", "M2_space"):
            assert results[cat].count == 0
            assert results[cat].markers == ()
