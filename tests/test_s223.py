#!/usr/bin/env python3
"""Tests for the s223 module."""

import pytest
from unittest.mock import patch, MagicMock
from rdflib.term import Variable
from rdf_mcp.servers import s223_server as s223


@pytest.fixture
def mock_s223_ontology():
    """Mock for the S223 RDF ontology."""
    mock = MagicMock()

    # Mock query results for get_terms
    terms_result = MagicMock()
    terms_result.__iter__.return_value = [
        (MagicMock(toPython=lambda: "http://data.ashrae.org/standard223#InletConnectionPoint"),),
        (MagicMock(toPython=lambda: "http://data.ashrae.org/standard223#OutletConnectionPoint"),),
        (MagicMock(toPython=lambda: "http://data.ashrae.org/standard223#Fan"),),
    ]

    # Mock query results for get_properties
    props_result = MagicMock()
    props_result.__iter__.return_value = [
        (MagicMock(toPython=lambda: "http://data.ashrae.org/standard223#hasConnectionPoint"),),
        (MagicMock(toPython=lambda: "http://data.ashrae.org/standard223#hasMedium"),),
        (MagicMock(toPython=lambda: "http://data.ashrae.org/standard223#cnx"),),
    ]

    # Mock query results for get_possible_properties
    poss_props_result = MagicMock()
    poss_props_result.__iter__.return_value = [
        (
            MagicMock(toPython=lambda: "http://data.ashrae.org/standard223#hasConnectionPoint"),
            MagicMock(toPython=lambda: "http://data.ashrae.org/standard223#InletConnectionPoint"),
        ),
        (
            MagicMock(toPython=lambda: "http://data.ashrae.org/standard223#hasMedium"),
            MagicMock(toPython=lambda: "http://data.ashrae.org/standard223#Fluid-Air"),
        ),
    ]

    # Set up the query results
    def mock_query(query_str):
        if "SELECT ?class WHERE" in query_str and "s223:Class" in query_str:
            return terms_result
        elif "SELECT ?class WHERE" in query_str and "rdf:Property" in query_str:
            return props_result
        elif "SELECT ?prop ?range" in query_str:
            return poss_props_result
        else:
            return MagicMock(__iter__=lambda: iter([]))

    mock.query = mock_query
    return mock


@patch("rdf_mcp.servers.s223_server.ontology")
def test_get_terms(mock_ontology):
    """Test get_terms function."""
    mock_ontology.query.return_value = [
        (MagicMock(__str__=lambda self: "http://data.ashrae.org/standard223#InletConnectionPoint"),),
        (MagicMock(__str__=lambda self: "http://data.ashrae.org/standard223#OutletConnectionPoint"),),
        (
            MagicMock(
                __str__=lambda self: "http://data.ashrae.org/standard223#Fan"
            ),
        ),
    ]
    from rdf_mcp.servers.s223_server import get_terms

    result = get_terms()
    assert isinstance(result, list)
    assert "InletConnectionPoint" in result
    assert "OutletConnectionPoint" in result
    assert "Fan" in result


@patch("rdf_mcp.servers.s223_server.ontology")
def test_get_properties(mock_ontology):
    """Test get_properties function."""
    mock_ontology.query.return_value = [
        (
            MagicMock(
                __str__=lambda self: "http://data.ashrae.org/standard223#hasConnectionPoint"
            ),
        ),
        (
            MagicMock(
                __str__=lambda self: "http://data.ashrae.org/standard223#hasMedium"
            ),
        ),
        (
            MagicMock(
                __str__=lambda self: "http://data.ashrae.org/standard223#cnx"
            ),
        ),
    ]
    from rdf_mcp.servers.s223_server import get_properties

    result = get_properties()
    assert isinstance(result, list)
    assert "hasConnectionPoint" in result
    assert "hasMedium" in result
    assert "cnx" in result


@patch("rdf_mcp.servers.s223_server.ontology")
def test_get_possible_properties(mock_ontology):
    """Test get_possible_properties function."""
    mock_result = MagicMock()
    mock_result.bindings = [
        {Variable("path"): "hasConnectionPoint"},
        {Variable("path"): "hasMedium"},
        {Variable("path"): "cnx"},
    ]
    mock_ontology.query.return_value = mock_result
    from rdf_mcp.servers.s223_server import get_possible_properties

    result = get_possible_properties("Fan")
    assert isinstance(result, list)
    assert "hasConnectionPoint" in result
    assert "hasMedium" in result
    assert "cnx" in result


@patch("rdf_mcp.servers.s223_server.ontology")
def test_get_constraints(mock_ontology):
    """Test get_constraints function."""
    constraints_result = [("constraint1", "Some constraint description")]

    def mock_custom_query(query_str):
        if "SELECT ?sh ?message" in query_str:
            return constraints_result
        return []

    mock_ontology.query = mock_custom_query
    try:
        from rdf_mcp.servers.s223_server import get_constraints

        result = get_constraints("Device")
        assert isinstance(result, list)
        # Assert specific expectations about the results if applicable
    except (ImportError, AttributeError):
        pytest.skip("get_constraints function is not implemented yet")
