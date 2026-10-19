"""Vercel entrypoint shim: the hosted dashboard runs the same FastAPI app as the laptop.

Vercel's FastAPI preset looks for an ``app`` object in app/index/server/main at the repository
root, in ``src/`` or in ``app/``. Its docs do not list ``api/``, where this project's app lives,
so this one import removes the ambiguity. Locally nothing changes: uvicorn keeps serving
``api.main:app``.
"""

from api.main import app  # noqa: F401
