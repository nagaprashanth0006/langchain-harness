from langchain_core.tools import Tool, StructuredTool
from langchain_core.tools import tool


from langchain_ollama import ChatOllama


from typing import TypedDict, Annotated, Any, Optional, List, Dict


from langgraph.checkpoint.memory import MemorySaver


from langgraph.prebuilt import create_react_agent


from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import START, END, MessagesState, StateGraph, add_messages
from requests import get
import os


from vector_store_operations import ensure_vectorstore, retrieve_context


MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@tool
def check_kb(prompt, collection_name="default"):
    """
    Takes prompt and performs similarity search on the vectorstore knowledge base
    :param prompt:
    :return: list of documents matching the similarity search
    """
    vs = ensure_vectorstore(collection_name=collection_name)
    context = retrieve_context(vs, prompt)
    return context


@tool
def http_requests(url):
    """
    Takes url as input and calls it using requests.get method
    :param url:
    :return: response of the get call
    """
    response = get(url, timeout=30)
    response.raise_for_status()
    return response.text


tools = [check_kb, http_requests]


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]


checkpointer = MemorySaver()


def chatbot(state: State) -> MessagesState:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, num_ctx=4092)
llm_with_tools = llm.bind_tools(tools)


def build_graph():
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")
    return builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ask the graph to search the local knowledge base."
    )
    parser.add_argument("question")
    args = parser.parse_args()
    result = build_graph().invoke(
        {"messages": [("user", args.question)]},
        config={"configurable": {"thread_id": "cli"}},
    )
    print(result["messages"][-1].content)
