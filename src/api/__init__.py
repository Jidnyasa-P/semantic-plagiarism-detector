"""
src/api module
--------------
REST API package providing endpoints for external LMS integrations (Canvas, Moodle, etc.).
"""

from src.api.app import app

__all__ = ["app"]
