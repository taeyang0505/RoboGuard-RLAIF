import os
import time
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, START, END

load_dotenv()

# State definition: includes retry tracking and evaluator feedback for RL loop
class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    feedback: str      # Evaluator feedback used as verbal policy signal
    pass_fail: str     # Verification result: PASS or FAIL
    retry_count: int   # Number of revision cycles completed

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- Node 1: Retrieve ---
def retrieve_node(state: AgentState):
    if state.get("retry_count", 0) == 0:
        print("[INFO] Retrieve: Querying Vector DB for relevant context.")
    docs = retriever.invoke(state["question"])
    return {"context": "\n\n".join([doc.page_content for doc in docs])}

# --- Node 2: Generate (Actor — base policy or revision policy) ---
def generate_node(state: AgentState):
    retry_cnt = state.get("retry_count", 0)

    if retry_cnt == 0:
        # Base policy: initial response from retrieved context only
        print("[INFO] Generate: Producing initial response (base policy).")
        prompt = (
            "Role: Technical documentation assistant for UR10e industrial robot systems.\n"
            "Task: Answer the question below using only the information in the reference document.\n"
            "Constraints:\n"
            "- Do not use any knowledge outside the reference document.\n"
            "- If the document does not contain relevant information, respond with: "
            "'The requested information is not available in the provided documentation.'\n\n"
            f"Reference Document:\n{state['context']}\n\nQuestion: {state['question']}\nAnswer:"
        )
    else:
        # Revision policy: incorporate evaluator feedback to correct prior response
        print(f"[INFO] Generate: Revising response based on evaluator feedback (retry {retry_cnt}).")
        prompt = (
            "Context: The previous response was flagged as containing unverified information.\n"
            f"Evaluator Feedback:\n{state['feedback']}\n\n"
            "Revision Instructions:\n"
            "- Address all issues identified in the evaluator feedback.\n"
            "- Use only information explicitly stated in the reference document below.\n"
            "- If the document does not contain relevant information, respond with: "
            "'The requested information is not available in the provided documentation.'\n\n"
            f"Reference Document:\n{state['context']}\n\nQuestion: {state['question']}\nRevised Answer:"
        )

    response = llm.invoke(prompt).content
    return {"answer": response, "retry_count": retry_cnt + 1}

# --- Node 3: Evaluate (Reward Model / LLM-as-a-judge) ---
def evaluate_node(state: AgentState):
    print("[INFO] Evaluate: Running fact-verification via LLM-as-a-judge.")
    eval_prompt = (
        "Task: Evaluate whether the response below is strictly grounded in the provided reference document.\n"
        "- Output [PASS] on the first line if the response contains only information explicitly stated "
        "in the reference document.\n"
        "- Output [FAIL] on the first line if the response contains any information not found in the "
        "reference document, including inferred, assumed, or externally sourced content.\n"
        "After the verdict, provide a concise rationale (1-2 sentences).\n\n"
        f"Reference Document:\n{state['context']}\n\nResponse:\n{state['answer']}\n\nEvaluation:"
    )

    eval_result = llm.invoke(eval_prompt).content
    pass_fail = "PASS" if "[PASS]" in str(eval_result).upper() else "FAIL"

    if pass_fail == "FAIL":
        print(f"[WARN] Evaluate: FAIL — Unverified content detected. Requesting revision.")
    else:
        print("[INFO] Evaluate: PASS — Response verified against source document.")

    return {"feedback": eval_result, "pass_fail": pass_fail}

# --- Router: RL loop control ---
def should_continue(state: AgentState):
    if state["pass_fail"] == "PASS":
        return "end"
    elif state["retry_count"] >= 3:
        print("[WARN] Router: Maximum retry limit (3) reached. Terminating with best available response.")
        return "end"
    else:
        time.sleep(2)  # Throttle to avoid API rate-limit errors
        return "retry"

# --- Graph assembly (cyclic: supports RL revision loop) ---
workflow = StateGraph(AgentState)  # type: ignore[type-var]
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("evaluate", evaluate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "evaluate")
workflow.add_conditional_edges("evaluate", should_continue, {"end": END, "retry": "generate"})

app = workflow.compile()

# --- Entry point ---
if __name__ == "__main__":
    print("-" * 75)
    print("[INFO] RoboGuard RLAIF — UR10e Query Agent with Revision Loop (v3)")
    print("-" * 75)

    test_question = (
        "Is it permissible to fully submerge the robot arm to a depth of 10 meters "
        "for underwater operations? Provide justification based on the IP rating "
        "specifications in the documentation."
    )

    print(f"\n[INFO] Input query: {test_question}")

    result = app.invoke({"question": test_question, "retry_count": 0})

    print("\n" + "-" * 75)
    print("[INFO] Final Response")
    print(result["answer"])
    print("-" * 75 + "\n")