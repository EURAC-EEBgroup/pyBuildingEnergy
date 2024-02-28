#!/usr/bin/env python

"""Tests for `pybuildingenergy` package."""

__author__ = "Daniele Antonucci"
__copyright__ = "Daniele Antonucci"
__license__ = "MIT"


import pytest


# try:
from pybuildingenergy import pybuildingenergy

# except ModuleNotFoundError:
    # import sys
    # sys.path.insert(1, '/home/osomova/Projects/vct/vctlib/src')
    # from vctlib.model import Building, ThermostaticalProperties, \
    #     BuildingCreateException
    # from vctlib.constant import VENT_RATES_MU


@pytest.fixture
def response(snapshot):
    """Sample .
    
    """
    
    result = {
        'ax':200
    }
    assert snapshot.assert_match(result, "test.json")
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


# def test_content(response):
#     """Sample pytest test function with the pytest fixture as an argument."""
#     # from bs4 import BeautifulSoup
#     # assert 'GitHub' in BeautifulSoup(response.content).title.string
