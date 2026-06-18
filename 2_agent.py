import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

# State definition
class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    hallucination_score: str

# Vector DB and LLM initialization
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- Node 1: Retrieve ---
def retrieve_node(state: AgentState):
    print("[INFO] Retrieve: Querying Vector DB for relevant context.")
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

# --- Node 2: Generate ---
def generate_node(state: AgentState):
    print("[INFO] Generate: Producing response from retrieved context.")
    prompt = (
        "Role: Technical documentation assistant for UR10e industrial robot systems.\n"
        "Task: Answer the question below using only the information in the reference document.\n"
        "Constraints: Do not use any knowledge outside the reference document. "
        "If the document does not contain relevant information, state that explicitly.\n\n"
        f"Reference Document:\n{state['context']}\n\nQuestion: {state['question']}\nAnswer:"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

# --- Node 3: Evaluate (LLM-as-a-judge) ---
def evaluate_node(state: AgentState):
    print("[INFO] Evaluate: Running fact-verification via LLM-as-a-judge.")
    eval_prompt = (
        "Task: Evaluate whether the response below is strictly grounded in the provided reference document.\n"
        "Output [PASS] if the response contains only information explicitly stated in the document.\n"
        "Output [FAIL] if the response contains any information not found in the document.\n\n"
        f"Reference Document:\n{state['context']}\n\nResponse:\n{state['answer']}\n\nEvaluation:"
    )
    eval_response = llm.invoke(eval_prompt)
    return {"hallucination_score": str(eval_response.content).strip()}

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

# --- Entry point ---
if __name__ == "__main__":
    print("-" * 50)
    print("[INFO] RoboGuard — UR10e Query Agent (v1, baseline)")
    print("-" * 50)

    test_question = "UR10e 로봇의 최대 적재량(Payload)은 정확히 몇 kg인가요?"
    print(f"\n[INFO] Input query: {test_question}")

    result = app.invoke({"question": test_question})

    print("\n" + "-" * 50)
    print("[INFO] Final Response")
    print(result["answer"])
    print("\n[INFO] Verification Result (LLM-as-a-judge)")
    print(f"       {result['hallucination_score']}")
    print("-" * 50 + "\n")