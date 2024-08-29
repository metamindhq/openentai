from litestar import Litestar, get
from modules.content.routes import content_router


@get("/")
async def index() -> dict[str, str]:
    """
    Return a simple system status response.
    """
    return {
        "status": "ok",
        "message": "System is running",
        "version": "0.0.1"
    }

app = Litestar(route_handlers=[index, content_router])