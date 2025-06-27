from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from graph.planner_graph import app as langgraph_app
from langchain_core.messages import HumanMessage

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/plan", response_class=HTMLResponse)
async def plan_trip(
    request: Request,
    city: str = Form(...),
    interests: str = Form(...)
):
    state = {
        "messages": [HumanMessage(content="Plan my trip")],
        "city": city,
        "interests": [i.strip() for i in interests.split(",")],
        "itinerary": "",
        "weather": "",
        "attractions": "",
        "info": "",
    }

    for s in langgraph_app.stream(state):
        state = s

    return templates.TemplateResponse("home.html", {
        "request": request,
        "city": city,
        "interests": interests,
        "itinerary": state.get("itinerary", "Trip planning failed. Please try again.")
    })
