#!/usr/bin/env python3
"""Tests for the brick module."""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import tempfile
from rdflib import Graph, URIRef, Namespace, BRICK, RDFS
from rdf_mcp.servers import brick_server as brick


class TestBrickModule:
    """Test suite for the brick module."""

    @pytest.fixture
    def setup_module_path(self):
        """Set up the module path for importing brick."""
        # Add the parent directory to sys.path so we can import brick
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        return parent_dir


@patch("rdf_mcp.servers.brick_server.ontology")
def test_expand_abbreviation(mock_ontology):
    """Test expand_abbreviation function."""
    from rdf_mcp.servers.brick_server import expand_abbreviation, CLASS_DICT
    from unittest.mock import patch

    with patch.dict(
        "rdf_mcp.servers.brick_server.CLASS_DICT",
        {"AHU": "Air_Handler_Unit", "VAV": "Variable_Air_Volume"},
    ):
        # Test AHU abbreviation
        result = expand_abbreviation("AHU")
        assert len(result) <= 5  # Should return no more than 5 results
        # Test that mock CLASS_DICT is used
        assert result  # Should return a non-empty list


@patch("rdf_mcp.servers.brick_server.ontology")
def test_get_terms(mock_ontology):
    """Test get_terms function."""
    mock_ontology.query.return_value = [
        (
            MagicMock(
                __str__=lambda self: "https://brickschema.org/schema/Brick#Air_Temperature_Sensor"
            ),
        ),
        (
            MagicMock(
                __str__=lambda self: "https://brickschema.org/schema/Brick#CO2_Level_Sensor"
            ),
        ),
        (
            MagicMock(
                __str__=lambda self: "https://brickschema.org/schema/Brick#Methane_Level_Sensor"
            ),
        ),
        (
            MagicMock(
                __str__=lambda self: "https://brickschema.org/schema/Brick#PM2.5_Sensor"
            ),
        ),
        (
            MagicMock(
                __str__=lambda self: "https://brickschema.org/schema/Brick#Building"
            ),
        ),
        (
            MagicMock(
                __str__=lambda self: "https://brickschema.org/schema/Brick#Floor"
            ),
        ),
        (
            MagicMock(
                __str__=lambda self: "https://brickschema.org/schema/Brick#Room"
            ),
        ),
        (
            MagicMock(
                __str__=lambda self: "https://brickschema.org/schema/Brick#Office_Kitchen"
            ),
        ),
        (
            MagicMock(
                __str__=lambda self: "https://brickschema.org/schema/Brick#Site"
            ),
        ),
    ]
    from rdf_mcp.servers.brick_server import get_terms

    result = get_terms()

    assert isinstance(result, list)
    assert "Air_Temperature_Sensor" in result
    assert "CO2_Level_Sensor" in result
    assert "Methane_Level_Sensor" in result
    assert "PM2.5_Sensor" in result
    assert "Building" in result
    assert "Floor" in result
    assert "Room" in result
    assert "Office_Kitchen" in result
    assert "Site" in result


@patch("rdf_mcp.servers.brick_server.ontology")
def test_get_properties(mock_ontology):
    """Test get_properties function."""
    mock_ontology.query.return_value = [
        (MagicMock(__str__=lambda self: "https://brickschema.org/schema/Brick#hasUnit"),),
        (MagicMock(__str__=lambda self: "https://brickschema.org/schema/Brick#isPointOf"),),
        (MagicMock(__str__=lambda self: "https://brickschema.org/schema/Brick#hasPart"),),
        (MagicMock(__str__=lambda self: "https://brickschema.org/schema/Brick#area"),),
        (MagicMock(__str__=lambda self: "https://brickschema.org/schema/Brick#value"),),
    ]
    from rdf_mcp.servers.brick_server import get_properties

    result = get_properties()

    assert isinstance(result, list)
    assert "hasUnit" in result
    assert "isPointOf" in result
    assert "hasPart" in result
    assert "area" in result
    assert "value" in result
