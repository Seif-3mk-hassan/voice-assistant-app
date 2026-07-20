# the_llm.py
from dotenv import load_dotenv
load_dotenv()
import os
import asyncio
api_key = os.environ["GROQ_API_KEY"]

from typing import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0)

mcp_client = MultiServerMCPClient({
    "gmail": {
        "command": "npx",
        "args": ["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
        "transport": "stdio",
    },
    "calendar": {
        "command": "npx",
        "args": ["-y", "@gongrzhe/server-calendar-autoauth-mcp"],
        "transport": "stdio",
    },
})

class voice_assistant(TypedDict):
    name: str
    question: str
    answer: str
    memory: list[dict]

async def build_agent():
    tools = await mcp_client.get_tools()
    return create_react_agent(llm, tools)

def make_assistant_node(agent):
    def assistant_node(state: voice_assistant) -> dict:
        messages = []
        for m in state["memory"]:
            messages.append(HumanMessage(content=m["question"]))
            messages.append(AIMessage(content=m["answer"]))
        messages.append(HumanMessage(content=state["question"]))

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                result = asyncio.run(agent.ainvoke({"messages": messages}))
                answer = result["messages"][-1].content
                return {"question": state["question"], "answer": answer}
            except Exception as e:
                if attempt == max_retries:
                    return {"question": state["question"], "answer": f"Error: {e}"}
    return assistant_node

def build_assistant_system():
    """Returns a compiled graph, ready to invoke. Call once, reuse across turns."""
    agent = asyncio.run(build_agent())
    builder = StateGraph(voice_assistant)
    builder.add_node("assistant", make_assistant_node(agent))
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)
    return builder.compile()