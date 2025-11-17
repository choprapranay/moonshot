import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(1, parent_dir)

from backend.infrastructure.supabase_client import SupabaseClient 

class ModelStorage:
    def __init__(self):
        self.client = SupabaseClient().client

    def upload_model(self, file_path: str, dest_path: str):
        with open(file_path, "rb") as f:
            res = self.client.storage.from_("models").upload(
                path=dest_path,
                file=f,
                file_options={"content-type": "application/octet-stream"}
            )
        return res