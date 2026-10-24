from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/api/ask")
def ask(request: QueryRequest):
    query = request.query

    answer = "Đã nhận query: " + query

    return {
        "answer": answer
    }