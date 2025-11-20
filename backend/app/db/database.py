from supabase import create_client
from app.core.config import settings
import threading

# Singleton pattern: client is created once and reused
# Thread-safe lazy initialization
_supabase = None
_lock = threading.Lock()

def get_supabase_client():
    global _supabase
    if _supabase is None:
        with _lock:  # Thread-safe: only one thread creates the client
            # Double-check pattern: another thread might have created it while we waited
            if _supabase is None:
                if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
                    raise RuntimeError("Missing Supabase credentials")
                _supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    return _supabase