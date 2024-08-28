from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
from datetime import datetime


class Content(BaseModel):
    id: str
    content: str
    embedding: np.ndarray
    policy: Optional[str] = Field(default="public")
    reference_link: Optional[str] = Field(default=None)
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())
    
    created_by: str
    updated_by: str
    
    metadata: dict
    
    version: int = Field(default=1)
    
    def create_table_ddl(cls) -> str:
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
    
    def create_index_ddl(cls) -> str:
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
    
    def get_insert_query_string(cls) -> str:
        query = """
        INSERT INTO content (id, content, embedding, policy, reference_link, created_at, updated_at, created_by, updated_by, metadata, version)
        VALUES (%(id)s, %(content)s, %(embedding)s, %(policy)s, %(reference_link)s, %(created_at)s, %(updated_at)s, %(created_by)s, %(updated_by)s, %(metadata)s, %(version)s)
        """
        return query
    
    def to_dict(self) -> dict:
        return self.model_dump()
        
        
    