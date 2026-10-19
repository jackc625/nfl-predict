"""NFL Prediction API Package.

This file imports nothing on purpose (Phase 36 D-11). It used to import ``.main``, so importing a
small module such as ``api.cache_schema`` or ``api.services`` from ``notify/`` built the whole
FastAPI app and loaded every dependency it pulls in. The app lives at ``api.main:app``, which is
what uvicorn, the Makefile, the runbook and the root ``index.py`` shim all use.
"""
