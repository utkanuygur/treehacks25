import openai
from typing import Annotated
from typing_extensions import TypedDict

# For typed messages rather than dict-based
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

###############################################################################
# 1. Set your OpenAI API key directly in the code
###############################################################################
openai.api_key = "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"

def call_openai(messages):
    """
    messages: List of dicts like { 'role': 'system'|'user'|'assistant', 'content': <string> }
    Returns the string content of the model's reply.
    
    NOTE: We are storing typed messages (AIMessage, HumanMessage, etc.) in 'state["messages"]'
    internally. However, for this simple call, we still build dict-based 'messages' to pass
    to openai.ChatCompletion.create().
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # or "gpt-4", etc.
        messages=messages,
        temperature=0.0
    )
    return response["choices"][0]["message"]["content"]

###############################################################################
# 2. Define the State for LangGraph
###############################################################################
class MyState(TypedDict):
    # We'll store typed messages (HumanMessage, AIMessage, etc.) in this list
    # add_messages means each node's returned messages get appended to the existing list
    messages: Annotated[list, add_messages]
    os_answer: str
    mobile_answer: str
    application_answer: str
    final_answer: str

graph_builder = StateGraph(MyState)

###############################################################################
# 3. Define Node Functions
###############################################################################

def orchestrator_initial(state: MyState) -> dict:
    """
    First orchestrator call:
    - We confirm the user's intent and plan to gather OS/Mobile/App answers.
    """
    # Grab the last message content. Because we are storing typed messages, use .content
    if state["messages"]:
        user_input = state["messages"][-1].content
    else:
        user_input = "No user input"
    
    prompt = (
        "You are the Orchestrator. The user said:\n\n"
        f"{user_input}\n\n"
        "Acknowledge the request. Next, we will gather OS, mobile, and application "
        "perspectives before providing a final answer."
    )
    ai_response = call_openai(
        [
            {"role": "system", "content": prompt}
        ]
    )
    # Return only the newly generated message as an AIMessage
    return {
        "messages": [AIMessage(content=ai_response)]
    }

def os_llm(state: MyState) -> dict:
    """
    Answers from an OS perspective.
    """
    # Here, we're referencing the *first* message's content (index 0),
    # which we assume is the user's original question.
    if state["messages"]:
        user_input = state["messages"][0].content
    else:
        user_input = "No user input"

    prompt = (
        "You are an OS expert. The user asked:\n\n"
        f"{user_input}\n\n"
        "Answer from an operating system perspective."
    )
    ai_response = call_openai(
        [
            {"role": "system", "content": prompt}
        ]
    )
    # We store the OS answer in the state. We do *not* append a new 'messages' entry
    return {"os_answer": ai_response}

def mobile_llm(state: MyState) -> dict:
    """
    Answers from a mobile device perspective.
    """
    if state["messages"]:
        user_input = state["messages"][0].content
    else:
        user_input = "No user input"

    prompt = (
        "You are a mobile device expert. The user asked:\n\n"
        f"{user_input}\n\n"
        "Answer from a mobile perspective."
    )
    ai_response = call_openai(
        [
            {"role": "system", "content": prompt}
        ]
    )
    return {"mobile_answer": ai_response}

def application_llm(state: MyState) -> dict:
    """
    Answers from an application perspective.
    """
    if state["messages"]:
        user_input = state["messages"][0].content
    else:
        user_input = "No user input"

    prompt = (
        "You are an application expert. The user asked:\n\n"
        f"{user_input}\n\n"
        "Answer from an application perspective."
    )
    ai_response = call_openai(
        [
            {"role": "system", "content": prompt}
        ]
    )
    return {"application_answer": ai_response}

def orchestrator_combine(state: MyState) -> dict:
    """
    Final orchestrator call: combine the three specialized answers.
    """
    os_ans = state.get("os_answer", "")
    mobile_ans = state.get("mobile_answer", "")
    app_ans = state.get("application_answer", "")

    prompt = (
        "You have the following three expert answers:\n\n"
        f"OS perspective:\n{os_ans}\n\n"
        f"Mobile perspective:\n{mobile_ans}\n\n"
        f"Application perspective:\n{app_ans}\n\n"
        "Combine them into a single, concise answer for the user."
    )
    ai_response = call_openai(
        [
            {"role": "system", "content": prompt}
        ]
    )
    return {"final_answer": ai_response}

###############################################################################
# 4. Add Nodes + Edges to the Graph
###############################################################################
graph_builder.add_node("orchestrator_initial", orchestrator_initial)
graph_builder.add_node("os_llm", os_llm)
graph_builder.add_node("mobile_llm", mobile_llm)
graph_builder.add_node("application_llm", application_llm)
graph_builder.add_node("orchestrator_combine", orchestrator_combine)

# A simple linear flow:
#   START -> orchestrator_initial -> os_llm -> mobile_llm -> application_llm -> orchestrator_combine -> END
graph_builder.add_edge(START, "orchestrator_initial")
graph_builder.add_edge("orchestrator_initial", "os_llm")
graph_builder.add_edge("os_llm", "mobile_llm")
graph_builder.add_edge("mobile_llm", "application_llm")
graph_builder.add_edge("application_llm", "orchestrator_combine")
graph_builder.add_edge("orchestrator_combine", END)

graph = graph_builder.compile()

###############################################################################
# 5. Run the Graph (Example)
###############################################################################
if __name__ == "__main__":
    user_query = "My Mac shows a strange error for 3 days. What could be causing it?"

    # Instead of using dict-based messages, let's store typed messages from the get-go:
    initial_state = {
        # We'll store a HumanMessage to represent the user
        "messages": [HumanMessage(content=user_query)],
        "os_answer": "",
        "mobile_answer": "",
        "application_answer": "",
        "final_answer": "",
    }

    # Since 'graph.run' no longer exists, we use graph.stream and capture the final state.
    for idx, event in enumerate(graph.stream(initial_state)):
        print(f"\n=== Event {idx} ===")
        print("State keys:", list(event.keys()))
        print(event)
