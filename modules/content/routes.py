from litestar import Router, get
from models.content import Content

@get("/")
async def get_content() -> list[Content]:
    return []

content_router = Router(path="/content", route_handlers=[get_content])