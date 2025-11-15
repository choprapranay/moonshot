import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

class SupabaseClient:

    def __init__(self):
        self.client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )