import gradio as gr
import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

class PlannerState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    country: str
    interests: List[str]
    days: int
    budget: str
    travel_type: str
    itinerary: str

# LLM Setup
llm = ChatGroq(
    temperature=0.4,
    groq_api_key=api_key,
    model_name="llama3-70b-8192"
)

# Prompt Template
itinerary_prompt = ChatPromptTemplate.from_template("""
You are an intelligent travel assistant.

Plan a {days}-day trip to {country} for a {travel_type} traveler with a {budget} budget.

Focus on these preferences: {interests}

The itinerary should be detailed, creative, and day-wise. Include:
- 3–4 key activities per day
- A mix of sightseeing, food, culture, and rest (if applicable)
""")

#  Main Planner Function
def input_fields(country: str, interests: List[str], days: int, budget: str, travel_type: str) -> str:
    state: PlannerState = {
        "messages": [],
        "country": country,
        "interests": interests,
        "days": days,
        "budget": budget,
        "travel_type": travel_type,
        "itinerary": ""
    }

    # Build prompt and call LLM
    formatted_prompt = itinerary_prompt.format_messages(
        country=country,
        interests=", ".join(interests),
        days=days,
        budget=budget,
        travel_type=travel_type
    )

    state["messages"].append(HumanMessage(content=f"{country}, {interests}, {days}, {budget}, {travel_type}"))
    response = llm.invoke(formatted_prompt)
    state["itinerary"] = response.content
    state["messages"].append(AIMessage(content=response.content))

    return response.content

#  Gradio UI
interface = gr.Interface(
    fn=input_fields,
    inputs=[
        gr.Dropdown(label="Select country", choices=["India", "France", "Japan", "USA", "Italy", "Australia"]),
        gr.CheckboxGroup(label="Your preferences", choices=[
            "Adventure", "Beaches", "Mountains", "Food", "Culture", "Shopping", "Nightlife", "Wildlife"
        ]),
        gr.Slider(label="Trip duration (days)", minimum=1, maximum=14, step=1, value=3),
        gr.Radio(label="Budget", choices=["Low", "Medium", "High"]),
        gr.Radio(label="Travel type", choices=["Solo", "Couple", "Family", "Friends"])
    ],
    outputs=gr.Textbox(label="📅 AI-Powered Itinerary"),
    title="🧳 Personalized Travel Planner",
    description="Select your travel preferences. Get a customized, day-wise itinerary powered by AI."
)

interface.launch()
