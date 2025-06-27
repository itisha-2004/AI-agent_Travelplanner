from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from tools.tool_definitions import get_weather, get_attractions
from dotenv import load_dotenv
from typing import Annotated, List, TypedDict
import os

load_dotenv()


class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "messages"]
    city: str
    interests: List[str]
    itinerary: str


llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

tools = [get_weather, get_attractions]
llm_with_tools = llm.bind_tools(tools)

def planner_llm(state: PlannerState) -> PlannerState:
    response = llm_with_tools.invoke({"messages": state["messages"]})
    return {
        **state,
        "messages": state["messages"] + [response]
    }


def route_tool_use(state: PlannerState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "generate"

def generate_itinerary(state: PlannerState) -> PlannerState:
    summary_prompt = f"""
    Create a one-day trip plan for {state['city']} based on interests: {', '.join(state['interests'])}.
    Include relevant weather and attraction info.
    """
    response = llm.invoke(summary_prompt)
    return {
        **state,
        "itinerary": response.content,
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }


workflow = StateGraph(PlannerState)
workflow.add_node("planner", planner_llm)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("generate", generate_itinerary)

workflow.set_entry_point("planner")

workflow.add_conditional_edges("planner", route_tool_use, {
    "tools": "tools",
    "generate": "generate"
})

workflow.add_edge("tools", "planner")
workflow.add_edge("generate", END)

app = workflow.compile()
