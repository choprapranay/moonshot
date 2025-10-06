from pydantic import BaseModel
from typing import Optional
from uuid import UUID

class PlayerCreate(BaseModel):
    first_name: str
    last_name: str
    position: Optional[str] = None
    team: Optional[str] = None

class PlayerUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    position: Optional[str] = None
    team: Optional[str] = None

class PlayerResponse(BaseModel):
    id: UUID
    first_name: str
    last_name: str
    position: Optional[str]
    team: Optional[str]
