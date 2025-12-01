import os
from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv

# Load .env from the backend directory
backend_dir = Path(__file__).resolve().parent.parent
env_path = backend_dir / ".env"
load_dotenv(dotenv_path=env_path)

class SupabaseClient:

    _instance = None


    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
            cls._instance.client = create_client(
            os.getenv("SUPABASE_URL"),
            supabase_key
        )
        return cls._instance
