import pytest

@pytest.fixture
def mock_gemini_response():
    """Fixture to provide a mocked Gemini API response for tests."""
    return {
        "text": "This is a mocked analysis response.",
        "category": "plant"
    }

@pytest.fixture
def sample_image_path():
    """Fixture to provide a dummy image path."""
    return "tests/data/sample_leaf.jpg"
