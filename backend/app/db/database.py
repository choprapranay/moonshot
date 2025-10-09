from supabase import create_client
from app.core.config import settings

_supabase = None

def get_supabase_client():
    global _supabase
    if _supabase is None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise RuntimeError("Missing Supabase credentials")
        _supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    return _supabase

supabase = get_supabase_client()