import os
from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv

# Load .env from the backend directory
backend_dir = Path(__file__).resolve().parent.parent
env_path = backend_dir / ".env"
load_dotenv(dotenv_path=env_path)

class SupabaseClient:

    def __init__(self):
        # Use service role key if available (for admin operations), otherwise anon key
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        self.client = create_client(
            os.getenv("SUPABASE_URL"),
            supabase_key
        )