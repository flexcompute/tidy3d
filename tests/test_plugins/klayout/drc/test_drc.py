"""Tests tidy3d/plugins/klayout/drc/drc.py"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

from pydantic import ValidationError
import pytest

import tidy3d as td
from tidy3d.exceptions import FileError
from tidy3d.plugins.klayout.drc.drc import DRCRunner
from tidy3d.plugins.klayout.drc.results import DRCResults, parse_violation_value
from tidy3d.plugins.klayout.util import check_installation

filepath = Path(os.path.dirname(os.path.abspath(__file__)))


def test_check_klayout_not_installed(monkeypatch):
    """check_installation raises when KLayout is not on PATH.

    Use monkeypatch to simulate absence, avoiding reliance on CI environment.
    """
    monkeypatch.setattr("tidy3d.plugins.klayout.util.which", lambda _cmd: None)
    with pytest.raises(RuntimeError):
        check_installation(raise_error=True)


def test_check_klayout_installed(monkeypatch):
    """check_installation returns a path and does not raise when present."""
    fake_path = "/usr/local/bin/klayout"
    monkeypatch.setattr("tidy3d.plugins.klayout.util.which", lambda _cmd: fake_path)
    assert check_installation(raise_error=True) == fake_path


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
        **to_gds_file_kwargs,
    ):
        """Calls DRCRunner.run with dummy run_drc_on_gds()"""

        # monkeypatch run_drc_on_gds() since the test machines do not have KLayout installed
        def mock_run_drc_on_gds(config):
            return DRCResults.load(filepath / "drc_results.lyrdb")

        monkeypatch.setattr("tidy3d.plugins.klayout.drc.drc.run_drc_on_gds", mock_run_drc_on_gds)

        runner = DRCRunner(drc_runset=drc_runsetfile, verbose=verbose)
        return runner.run(
            source=source,
            td_object_gds_savefile=td_object_gds_savefile,
            resultsfile=resultsfile,
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
            resultsfile=filepath / "drc_results.lyrdb",
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
            resultsfile=filepath / "drc_results.lyrdb",
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
                resultsfile=filepath / "drc_results.lyrdb",
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
                resultsfile=filepath / "drc_results.lyrdb",
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
                resultsfile=filepath / "drc_results.lyrdb",
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
                resultsfile=filepath / "drc_results.lyrdb",
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
                resultsfile=filepath / "drc_results.lyrdb",
            )


class TestDRCResults:
    """Test DRCResults"""

    @pytest.fixture(scope="class")
    def drc_results(self):
        """Load the DRC results"""
        return DRCResults.load(filepath / "drc_results.lyrdb")

    @pytest.fixture(scope="class")
    def drc_results_widthonly_clean(self):
        """Load the DRC results"""
        return DRCResults.load(filepath / "drc_results_widthonly_clean.lyrdb")

    def test_result_file_load(self, tmp_path):
        """Test that result file loading works"""
        # this should not raise an error
        DRCResults.load(filepath / "drc_results.lyrdb")

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

    def test_is_drc_clean(self, drc_results, drc_results_widthonly_clean):
        """Test DRCResults.is_clean"""
        assert not drc_results.is_clean
        assert drc_results_widthonly_clean.is_clean

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
        assert len(drc_results["min_area"].markers[0].polygons) == 2
        assert drc_results["min_area"].markers[0].polygons[0] == (
            (-0.6, -0.112),
            (-0.6, 0.555),
            (0.217, 0.555),
            (0.217, -0.112),
        )
        assert drc_results["min_area"].markers[0].polygons[1] == (
            (-0.31, 0.24),
            (-0.206, 0.24),
            (-0.206, 0.342),
            (-0.31, 0.342),
        )
        assert len(drc_results["min_hole"].markers[0].polygons) == 1
        assert drc_results["min_hole"].markers[0].polygons[0] == (
            (-0.31, 0.24),
            (-0.31, 0.342),
            (-0.206, 0.342),
            (-0.206, 0.24),
        )

    @pytest.mark.parametrize(
        "edge_value, expected_edge",
        [
            ("edge: (1.0,2.0;3.0,4.0)", ((1.0, 2.0), (3.0, 4.0))),
            ("edge: (-1.0,-2.0;-3.0,-4.0)", ((-1.0, -2.0), (-3.0, -4.0))),
        ],
    )
    def test_parse_edge(self, edge_value, expected_edge):
        """Test parsing edge violation values."""
        edge_result = parse_violation_value(edge_value)
        assert edge_result.edge == expected_edge

    def test_parse_edge_pair(self):
        """Test parsing edge-pair violation values."""
        edge_pair_value = "edge-pair: (1.0,2.0;3.0,4.0)|(5.0,6.0;7.0,8.0)"
        edge_pair_result = parse_violation_value(edge_pair_value)
        assert edge_pair_result.edge_pair[0] == ((1.0, 2.0), (3.0, 4.0))
        assert edge_pair_result.edge_pair[1] == ((5.0, 6.0), (7.0, 8.0))

    def test_parse_polygon(self):
        """Test parsing a single polygon violation string."""
        polygon_value = "polygon: (1.0,2.0;3.0,4.0;5.0,6.0;1.0,2.0)"
        polygon_result = parse_violation_value(polygon_value)
        assert polygon_result.polygons[0] == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (1.0, 2.0))

    def test_parse_multiple_polygons(self):
        """Test parsing multiple polygons violation string."""
        polygon_value = (
            "polygon: (1.0,2.0;3.0,4.0;5.0,6.0;1.0,2.0/7.0,8.0;9.0,10.0;11.0,12.0;7.0,8.0)"
        )
        polygon_result = parse_violation_value(polygon_value)
        assert polygon_result.polygons[0] == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (1.0, 2.0))
        assert polygon_result.polygons[1] == ((7.0, 8.0), (9.0, 10.0), (11.0, 12.0), (7.0, 8.0))

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
            parse_violation_value(invalid_edge)

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
            parse_violation_value(invalid_edge_pair)

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
            parse_violation_value(invalid_polygon)

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
            parse_violation_value(invalid_polygons)

    def test_parse_violation_value_unknown_type(self):
        """Test parsing unknown violation type."""
        with pytest.raises(ValueError):
            parse_violation_value("unknown: (1.0,2.0)")
