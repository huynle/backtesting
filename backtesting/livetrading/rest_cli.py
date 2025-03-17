import json
import logging
from typing import Optional
from urllib.parse import urljoin

import requests
from requests import ConnectionError

logger = logging.getLogger(__name__)


class RestClient:
    """ "Class for REST API.
    :param config: Config settings for exchange.
    """

    def __init__(self, config):
        self.url = config["api_url"]
        self.session = requests.Session()
        self.session.auth = (config.get("username"), config.get("password"))

    def call(self, method, apipath, params: Optional[dict] = None, data=None):
        if str(method).upper() not in ("GET", "POST", "PUT", "DELETE"):
            raise ValueError(f"invalid method <{method}>")

        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        url = urljoin(self.url, apipath)

        try:
            resp = self.session.request(
                method, url, headers=headers, data=json.dumps(data), params=params
            )
            if resp.status_code == 200:
                return resp.json()
            return resp.text
        except ConnectionError:
            logger.warning("Connection error")


class MockRestClient:
    """
    Mock class for REST API to simulate API calls without actual network requests.

    This class is designed for testing purposes, allowing you to define expected
    responses for specific API endpoints. It removes the need for actual network
    communication, making tests faster and more reliable.
    """

    def __init__(self, config):
        """
        Initializes the MockRestClient with a configuration dictionary.

        :param config: A dictionary containing configuration settings, including 'api_url'.
        """
        self.url = None
        self.mock_responses = {}  # type: dict

    def add_mock_response(self, method: str, apipath: str, response: dict):
        """
        Adds a mock response for a specific API endpoint.

        :param method: The HTTP method (e.g., 'GET', 'POST').
        :param apipath: The API endpoint path (e.g., '/products/BTC-USD').
        :param response: The mock response data (dictionary).
        """
        key = (method.upper(), apipath)
        self.mock_responses[key] = response

    def call(self, method: str, apipath: str, params: Optional[dict] = None, data=None):
        """
        Simulates an API call and returns the mock response.

        :param method: The HTTP method (e.g., 'GET', 'POST').
        :param apipath: The API endpoint path (e.g., '/products/BTC-USD').
        :param params: Optional dictionary of query parameters.
        :param data: Optional data to send in the request body.

        :raises ValueError: If the HTTP method is invalid.
        :raises KeyError: If no mock response is found for the given method and apipath.

        :return: The mock response data (dictionary).
        """
        if str(method).upper() not in ("GET", "POST", "PUT", "DELETE"):
            raise ValueError(f"invalid method <{method}>")

        key = (method.upper(), apipath)
        if key not in self.mock_responses:
            raise KeyError(f"No mock response found for {method} {apipath}")

        return self.mock_responses[key]
