from pydantic import BaseModel
from typing import Optional, List


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer:str

