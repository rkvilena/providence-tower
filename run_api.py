from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that "import core.*" works
# regardless of how the script is launched.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.api.main import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "run_api:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info",
    )
