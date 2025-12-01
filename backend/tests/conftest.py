"""
Pytest configuration - mocks supabase before any test imports.
"""
import sys
from unittest.mock import MagicMock

if 'supabase' not in sys.modules:
    sys.modules['supabase'] = MagicMock()
    sys.modules['supabase'].create_client = MagicMock()

if 'dotenv' not in sys.modules:
    sys.modules['dotenv'] = MagicMock()
    sys.modules['dotenv'].load_dotenv = MagicMock()

