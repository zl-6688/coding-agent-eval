"""Pytest fixtures for database-backed integration tests.

Use these fixtures in tests for OrderRepo and any other DB-touching code.
They spin up a real (temporary) SQLite database — no mocking.
"""
import pytest

from orders import OrderRepo


@pytest.fixture
def db_path(tmp_path):
    """Absolute path to an isolated, empty SQLite file for one test run."""
    return str(tmp_path / "test_orders.db")


@pytest.fixture
def order_repo(db_path):
    """A fully initialised OrderRepo backed by a fresh test database."""
    repo = OrderRepo(db_path)
    repo.create_schema()
    return repo
