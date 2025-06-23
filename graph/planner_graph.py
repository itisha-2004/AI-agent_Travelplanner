from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tools.attraction_tool import get_attractions
from tools.whether_tool import get_weather
from rag.vector_store import retrieve_info
from dotenv import load_dotenv
from typing import Annotated, List, TypedDict
import os

load_dotenv()

class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "messages"]
    city: str
    interests: List[str]
    itinerary: str
    weather: str
    attractions: str
    info: str

llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant planning a trip to {city}. Consider interests: {interests}, weather: {weather}, attractions: {attractions}, and destination info: {info}. Provide a bullet-point itinerary."),
    ("human", "Plan my day trip.")
])

def get_weather_info(state: PlannerState) -> PlannerState:
    weather = get_weather(state["city"])
    return {**state, "weather": weather}

def get_attraction_info(state: PlannerState) -> PlannerState:
    attractions = get_attractions(state["city"], state["interests"])
    return {**state, "attractions": attractions}

def get_destination_info(state: PlannerState) -> PlannerState:
    info = retrieve_info(state["interests"])
    return {**state, "info": info}

def generate_itinerary(state: PlannerState) -> PlannerState:
    response = llm.invoke(prompt.format_messages(
        city=state["city"],
        interests=", ".join(state["interests"]),
        weather=state["weather"],
        attractions=state["attractions"],
        info=state["info"]
    ))
    return {**state, "itinerary": response.content, "messages": state["messages"] + [AIMessage(content=response.content)]}

# Graph workflow
workflow = StateGraph(PlannerState)
workflow.add_node("fetch_weather", get_weather_info)
workflow.add_node("fetch_attractions", get_attraction_info)
workflow.add_node("destination_info", get_destination_info)
workflow.add_node("generate_itinerary", generate_itinerary)

workflow.set_entry_point("fetch_weather")
workflow.add_edge("fetch_weather", "fetch_attractions")
workflow.add_edge("fetch_attractions", "destination_info")
workflow.add_edge("destination_info", "generate_itinerary")
workflow.add_edge("generate_itinerary", END)

app = workflow.compile()

