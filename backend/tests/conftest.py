"""
Pytest configuration - mocks supabase before any test imports.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

backend_dir = Path(__file__).parent.parent
neuralnets_dir = backend_dir / 'neuralnetsCA'
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(neuralnets_dir))

if 'supabase' not in sys.modules:
    sys.modules['supabase'] = MagicMock()
    sys.modules['supabase'].create_client = MagicMock()

if 'dotenv' not in sys.modules:
    sys.modules['dotenv'] = MagicMock()
    sys.modules['dotenv'].load_dotenv = MagicMock()

