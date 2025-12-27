"""Pytest configuration and fixtures for DailyPost MCP tests."""
import pytest
from unittest.mock import MagicMock
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def mock_weaviate_client():
    """Create a mock Weaviate client for testing."""
    mock_client = MagicMock()
    mock_client.is_ready.return_value = True
    mock_client.collections.get.return_value = MagicMock()
    return mock_client

@pytest.fixture
def sample_articles():
    """Create sample article data for testing."""
    return [
        {
            "id": "article_1",
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of AI...",
            "category": "AI",
            "date": "2025-01-15",
        },
        {
            "id": "article_2",
            "title": "Python Best Practices",
            "content": "Writing clean Python code...",
            "category": "Python",
            "date": "2025-01-10",
        },
        {
            "id": "article_3",
            "title": "Vector Databases Explained",
            "content": "Vector databases store embeddings...",
            "category": "Database",
            "date": "2025-01-20",
        },
    ]

@pytest.fixture
def mock_search_results():
    return [
        {"id": "article_1", "title": "Intro to ML", "similarity": 0.95, "rank": 1},
        {"id": "article_2", "title": "Python Tips", "similarity": 0.87, "rank": 2},
    ]

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "slow: slow tests")
