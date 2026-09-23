from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph


class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class OverallState(InputState, OutputState):
    pass

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

def llm_node(state: InputState):
    messages = [("system","你是一位乐于助人的智能小助理"),
                ("human",state["question"])]
    llm = ChatOpenAI(model="openrouter/free",
                     base_url="https://openrouter.ai/api/v1",
                     api_key="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",
                     temperature=0)
    response = llm.invoke(messages)
    return {"answer": response.content}

builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
builder.add_node("llm_node", llm_node)
builder.add_edge(START, "llm_node")
builder.add_edge(("llm_node"), END)

graph = builder.compile()
final_answer = graph.invoke({"question": "你是干什么吃的"})
print(final_answer["answer"])
