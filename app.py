from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from agent import build_agent
import uvicorn


app = FastAPI()

# Serve static files (for HTML frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

agent = build_agent()

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")
    chat_history = data.get("chat_history", [])
    final_state = agent.invoke({
        "question": question,
        "chat_history": chat_history
    })
    final_answer = final_state["chat_history"][-1].content
    return JSONResponse({"answer": final_answer})

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

