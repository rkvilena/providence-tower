from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that "import core.*" works
# regardless of how the script is launched.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from fastapi.staticfiles import StaticFiles
from core.api.main import create_app

app = create_app()
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "core.main:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info",
    )
