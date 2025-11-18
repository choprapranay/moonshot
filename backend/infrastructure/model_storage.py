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
    
    def download_model(self, source_path: str, dest_path: str):
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        res = self.client.storage.from_("models").download(source_path)
        
        with open(dest_path, "wb") as f:
            f.write(res)
        
        print(f"Downloaded model from {source_path} to {dest_path}")
        return dest_path
    
    def model_exists(self, path: str) -> bool:
        try:
            files = self.client.storage.from_("models").list(os.path.dirname(path))
            filename = os.path.basename(path)
            return any(f['name'] == filename for f in files)
        except:
            return False