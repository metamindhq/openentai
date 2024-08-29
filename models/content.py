from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
import numpy as np
from datetime import datetime
import psycopg2


class Content(BaseModel):
    id: str
    content: str
    embedding: np.ndarray
    policy: Optional[str] = None
    reference_link: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    created_by: str
    updated_by: str
    
    metadata: Optional[dict] = None
    
    version: int = Field(default=1)
    
    model_config  = ConfigDict(arbitrary_types_allowed=True)
    
    @staticmethod
    def create_table_ddl() -> str:
        """
        Generate an SQL CREATE TABLE query string for the Content model.

        Returns:
            str: An SQL CREATE TABLE query string.
        """
        query = """
        CREATE TABLE IF NOT EXISTS content (
            id VARCHAR(255) PRIMARY KEY,
            content TEXT,
            embedding VECTOR(0),
            policy VARCHAR(255),
            reference_link VARCHAR(510),
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            created_by VARCHAR(255),
            updated_by VARCHAR(255),
            metadata JSONB,
            version INT
        )
        """
        return query
    
    @staticmethod
    def create_index_ddl() -> str:
        """
        Generate an SQL CREATE INDEX query string for the Content model.

        Returns:
            str: An SQL CREATE INDEX query string.
        """
        query = """
        CREATE INDEX IF NOT EXISTS document_content_idx ON content (content) USING gin (to_tsvector('english', content));
        CREATE INDEX IF NOT EXISTS document_embedding_idx ON content USING diskann (embedding);
        CREATE INDEX IF NOT EXISTS document_policy_idx ON content (policy);
        CREATE INDEX IF NOT EXISTS document_reference_link_idx ON content (reference_link);
        CREATE INDEX IF NOT EXISTS document_created_by_idx ON content (created_by);
        CREATE INDEX IF NOT EXISTS document_updated_by_idx ON content (updated_by);
        CREATE INDEX IF NOT EXISTS document_metadata_idx ON content USING gin (metadata);
        CREATE INDEX IF NOT EXISTS document_version_idx ON content (version);
        """
        return query
    
    @staticmethod
    def get_insert_query_string() -> str:
        """
        Generate an SQL INSERT query string for the Content model.

        Returns:
            str: An SQL INSERT query string.
        """
        query = """
        INSERT INTO content (id, content, embedding, policy, reference_link, created_at, updated_at, created_by, updated_by, metadata, version)
        VALUES (%(id)s, %(content)s, %(embedding)s, %(policy)s, %(reference_link)s, %(created_at)s, %(updated_at)s, %(created_by)s, %(updated_by)s, %(metadata)s, %(version)s)
        """
        return query
    
    def get_update_query_string(self) -> str:
        """
        Generate an SQL UPDATE query string for the Content model.

        Returns:
            str: An SQL UPDATE query string.
        """
        query = """
        UPDATE content
        SET
            {% if content is not none %}content = %(content)s,{% endif %}
            {% if embedding is not none %}embedding = %(embedding)s,{% endif %}
            {% if policy is not none %}policy = %(policy)s,{% endif %}
            {% if reference_link is not none %}reference_link = %(reference_link)s,{% endif %}
            {% if updated_at is not none %}updated_at = %(updated_at)s,{% endif %}
            {% if updated_by is not none %}updated_by = %(updated_by)s,{% endif %}
            {% if metadata is not none %}metadata = %(metadata)s,{% endif %}
            {% if version is not none and version > 1 %}version = %(version)s,{% endif %}
            id = %(id)s  -- Ensure at least one SET clause
        WHERE id = %(id)s
        """
        return query.strip().rstrip(',')  # Remove trailing comma if present
    
    def to_dict(self) -> dict:
        """
        Convert the Content model to a dictionary.

        Returns:
            dict: A dictionary representation of the Content model.
        """
        return self.model_dump()
    
    @staticmethod
    def from_row(row: dict) -> 'Content':
        """
        Convert a psycopg2 row response to a Content model.
        
        Args:
            row (dict): A dictionary representing a database row.
        
        Returns:
            Content: An instance of the Content model.
        """
        return Content(
            id=row.get('id'),
            content=row.get('content'),
            embedding=row.get('embedding'),
            policy=row.get('policy'),
            reference_link=row.get('reference_link'),
            created_at=row.get('created_at'),
            updated_at=row.get('updated_at'),
            created_by=row.get('created_by'),
            updated_by=row.get('updated_by'),
            metadata=row.get('metadata'),
            version=row.get('version')
        )
        
        
    