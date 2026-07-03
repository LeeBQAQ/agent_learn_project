from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_HTML_PATH = Path(__file__).parent.parent.parent.parent / "static" / "chat.html"


@router.get("/", response_class=HTMLResponse)
def chat_page():
    return HTMLResponse(content=_HTML_PATH.read_text(encoding="utf-8"))
