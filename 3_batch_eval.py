import os
import csv
import time
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, START, END

load_dotenv()

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    hallucination_score: str

# Retrieve k=5 chunks for broader context coverage
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- Node 1: Retrieve ---
def retrieve_node(state: AgentState):
    docs = retriever.invoke(state["question"])
    return {"context": "\n\n".join([doc.page_content for doc in docs])}

# --- Node 2: Generate ---
def generate_node(state: AgentState):
    prompt = (
        "Role: Technical documentation assistant for UR10e industrial robot systems.\n"
        "Task: Answer the question below using only the information in the reference document.\n"
        "Constraints:\n"
        "- Do not use any knowledge outside the reference document.\n"
        "- If the document does not contain relevant information, respond with: "
        "'The requested information is not available in the provided documentation.'\n\n"
        f"Reference Document:\n{state['context']}\n\nQuestion: {state['question']}\nAnswer:"
    )
    return {"answer": llm.invoke(prompt).content}

# --- Node 3: Evaluate (LLM-as-a-judge) ---
def evaluate_node(state: AgentState):
    eval_prompt = (
        "Task: Evaluate whether the response below is strictly grounded in the provided reference document.\n"
        "Output [PASS] on the first line if the response contains only information explicitly stated in the document.\n"
        "Output [FAIL] on the first line if the response contains any information not found in the document, "
        "including inferred, assumed, or externally sourced content. "
        "After the verdict, provide a concise rationale (1-2 sentences).\n\n"
        f"Reference Document:\n{state['context']}\n\nResponse:\n{state['answer']}\n\nEvaluation:"
    )
    return {"hallucination_score": str(llm.invoke(eval_prompt).content).strip()}

# --- Pipeline assembly ---
workflow = StateGraph(AgentState)  # type: ignore[type-var]
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "evaluate")
workflow.add_edge("evaluate", END)
app = workflow.compile()

# --- Entry point: Batch evaluation pipeline ---
if __name__ == "__main__":
    print("-" * 75)
    print("[INFO] RoboGuard — Batch Evaluation Pipeline (v2, baseline)")
    print("-" * 75)

    test_questions = [
        "Q1 [Fault Recovery] A Protective Stop has been triggered due to joint overload. "
        "Describe the step-by-step procedure to clear the condition and resume operation using the Teach Pendant.",

        "Q2 [Numerical Specification] What is the maximum allowable current output (in amperes) "
        "provided by the 24V power supply on the I/O terminal block inside the Control Box?",

        "Q3 [Out-of-Scope Boundary Test] Is it permissible to fully submerge the robot arm to a depth "
        "of 10 meters for underwater operations? Provide justification based on the IP rating "
        "specifications in the documentation.",
    ]

    with open("eval_report.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Response", "Verification Result (LLM-as-a-judge)"])

        for q in test_questions:
            print(f"\n[INFO] Processing: {q[:80]}...")
            start = time.time()

            result = app.invoke({"question": q})
            elapsed = time.time() - start

            print(f"       Response preview: {result['answer'][:150]}...")
            print(f"       Verification: {result['hallucination_score']}")
            print(f"       Elapsed: {elapsed:.2f}s")

            writer.writerow([q, result['answer'], result['hallucination_score']])
            time.sleep(2)  # Throttle to avoid API rate-limit errors

    print("\n" + "-" * 75)
    print("[INFO] Batch evaluation complete. Results saved to 'eval_report.csv'.")
    print("-" * 75)