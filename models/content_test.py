import pytest
import numpy as np
from datetime import datetime
from models.content import Content

@pytest.fixture
def sample_content():
    return Content(
        id="test_id",
        content="Test content",
        embedding=np.array([0.1, 0.2, 0.3]),
        policy="test_policy",
        reference_link="https://example.com",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        created_by="test_user",
        updated_by="test_user",
        metadata={"key": "value"},
        version=1
    )

def test_create_table_ddl():
    ddl = Content.create_table_ddl()
    assert "CREATE TABLE IF NOT EXISTS content" in ddl
    assert "id VARCHAR(255) PRIMARY KEY" in ddl
    assert "embedding VECTOR(0)" in ddl
    assert "metadata JSONB" in ddl

def test_create_index_ddl():
    ddl = Content.create_index_ddl()
    assert "CREATE INDEX IF NOT EXISTS document_content_idx" in ddl
    assert "CREATE INDEX IF NOT EXISTS document_embedding_idx" in ddl
    assert "CREATE INDEX IF NOT EXISTS document_metadata_idx" in ddl

def test_get_insert_query_string():
    query = Content.get_insert_query_string()
    assert "INSERT INTO content" in query
    assert "VALUES (%(id)s, %(content)s, %(embedding)s" in query

def test_get_update_query_string(sample_content):
    query = sample_content.get_update_query_string()
    assert "UPDATE content" in query
    assert "SET" in query
    assert "WHERE id = %(id)s" in query

def test_to_dict(sample_content):
    content_dict = sample_content.to_dict()
    assert isinstance(content_dict, dict)
    assert content_dict['id'] == "test_id"
    assert content_dict['content'] == "Test content"
    assert np.array_equal(content_dict['embedding'], np.array([0.1, 0.2, 0.3]))
    assert content_dict['policy'] == "test_policy"
    assert content_dict['reference_link'] == "https://example.com"
    assert isinstance(content_dict['created_at'], datetime)
    assert isinstance(content_dict['updated_at'], datetime)
    assert content_dict['created_by'] == "test_user"
    assert content_dict['updated_by'] == "test_user"
    assert content_dict['metadata'] == {"key": "value"}
    assert content_dict['version'] == 1

def test_content_model_validation():
    with pytest.raises(ValueError):
        Content(
            id="test_id",
            content="Test content",
            embedding="invalid_embedding",  # Should be np.ndarray
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="test_user",
            updated_by="test_user"
        )

def test_content_model_optional_fields():
    content = Content(
        id="test_id",
        content="Test content",
        embedding=np.array([0.1, 0.2, 0.3]),
        created_at=datetime.now(),
        updated_at=datetime.now(),
        created_by="test_user",
        updated_by="test_user"
    )
    assert content.policy is None
    assert content.reference_link is None
    assert content.metadata is None
    assert content.version == 1  # Default value
