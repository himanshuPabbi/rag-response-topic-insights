import os
import time
import json
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.callbacks import BaseCallbackHandler

# --- NEW: Import Groq API Exception for targeted error handling ---
from groq import APIStatusError as GroqAPIError
# =========================================
# CONFIG
# =========================================
load_dotenv()

DATA_PATH = "IndicLegalQA Dataset_10K.json"
BASE_CHROMA_DIR = "./chroma_store_legal"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Models for evaluation
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768"
]

# Ablation parameters
CHUNK_SIZES = [300, 1200]
CHUNK_OVERLAP = 200
RETRIEVER_K_VALUES = [4, 8]

# --- NEW CONFIGURATION ---
BENCHMARK_SAMPLE_SIZE = 300  # Number of random questions to use for the benchmark
# -------------------------

# --- RATE LIMIT CONFIGURATION ---
MAX_RETRIES = 5
BASE_DELAY = 5 # Initial wait time in seconds
# --------------------------------

OUTPUT_CSV = "research_benchmark_full.csv"

# Deterministic seeds
random.seed(42)
np.random.seed(42)


# =========================================
# PROMPT (Unchanged)
# =========================================
LEGAL_PROMPT_TEMPLATE = """
You are a highly analytical Legal Assistant. Answer strictly using the provided context.

Context:
{context}

Question: {question}

Provide:
1. Legal Issue
2. Authority
3. Reasoning
4. Final Answer
"""

LEGAL_PROMPT = PromptTemplate(
    template=LEGAL_PROMPT_TEMPLATE,
    input_variables=["question", "context"]
)


# =========================================
# VECTOR DB BUILDER (Unchanged)
# =========================================
def build_or_load_vector_db(chunk_size: int):
    """
    Each chunk size gets its own Chroma DB.
    """
    vdb_path = f"{BASE_CHROMA_DIR}_{chunk_size}"
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(vdb_path):
        print(f"[OK] Loaded Chroma Store for chunk_size={chunk_size}")
        return Chroma(persist_directory=vdb_path, embedding_function=embeddings)

    print(f"[BUILD] Creating Chroma DB (chunk_size={chunk_size})")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    docs = [
        Document(
            page_content=item.get("answer", ""),
            metadata={
                "case_name": item.get("case_name", ""),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "category": item.get("category", ""),
                "judgment_date": item.get("judgment_date", "")
            },
        )
        for item in corpus
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=vdb_path
    )

    print(f"[DONE] Vector DB created → {len(chunks)} chunks")
    return vectordb


# =========================================
# TTFT CALLBACK (Unchanged)
# =========================================
class TTFTCallback(BaseCallbackHandler):
    def __init__(self):
        self.first_token_time = None

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if self.first_token_time is None:
            self.first_token_time = time.time()


# =========================================
# INIT RAG (Unchanged)
# =========================================
def init_rag(model_name: str, vdb, retriever_k: int):
    # Note: LangChain's ChatGroq client has a 'max_retries' parameter,
    # but implementing it manually here provides more granular control over backoff.
    callback = TTFTCallback()

    llm = ChatGroq(
        model_name=model_name,
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY"),
        streaming=True,
        callbacks=[callback]
    )

    retriever = vdb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": retriever_k}
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": LEGAL_PROMPT},
        return_source_documents=True
    )

    return chain, callback


# =========================================
# RETRIEVAL METRICS (Unchanged)
# =========================================
emb_eval = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def cosine_similarity(a, b):
    # Using 1e-9 for numerical stability, as in the original code
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def compute_retrieval_metrics(gt_answer: str, retrieved_docs: List[Document]):
    if not retrieved_docs:
        return 0, 0, 0, 0, False

    gt_embed = emb_eval.embed_query(gt_answer)
    retrieved_embeds = [emb_eval.embed_query(doc.page_content) for doc in retrieved_docs]

    sims = [cosine_similarity(gt_embed, r) for r in retrieved_embeds]

    threshold = 0.55
    precision = sum(s >= threshold for s in sims) / len(sims)
    recall = precision 

    max_sim = max(sims)
    rank = sims.index(max_sim) + 1
    mrr = 1.0 / rank
    ndcg = max_sim
    hit = max_sim >= threshold

    return precision, recall, mrr, ndcg, hit


# =========================================
# BENCHMARK (Modified with Retry Logic)
# =========================================
def run_benchmark(dataset: List[Dict[str, str]]):
    """
    Runs the full RAG benchmark on the provided dataset, handling Groq rate limits.
    """
    write_header = not os.path.exists(OUTPUT_CSV)

    print(f"📊 Running benchmark on a sample of {len(dataset)} entries.")

    for chunk_size in CHUNK_SIZES:
        vdb = build_or_load_vector_db(chunk_size)
        if vdb is None:
            continue

        for retriever_k in RETRIEVER_K_VALUES:
            for model_name in GROQ_MODELS:

                print(
                    f"\n=== BENCHMARKING ===\n"
                    f"Model: {model_name}\n"
                    f"Chunk Size: {chunk_size}\n"
                    f"Retriever K: {retriever_k}\n"
                )

                qa_chain, callback = init_rag(model_name, vdb, retriever_k)

                for entry in dataset:
                    q = entry["question"]
                    a = entry["answer"]
                    
                    result = None
                    
                    # --- Exponential Backoff Retry Loop ---
                    for attempt in range(MAX_RETRIES):
                        t_start = time.time()
                        callback.first_token_time = None
                        
                        try:
                            # Attempt the API call
                            result = qa_chain.invoke({"query": q})
                            break # Success! Break the retry loop
                        
                        except Exception as e:
                            # GroqAPIError is a specific class, but a generic Exception 
                            # check with a message check is often more robust in chain environments.
                            error_message = str(e)

                            # Check for 429 rate limit or related messages
                            if "429" in error_message or "rate limit" in error_message.lower():
                                # Calculate Exponential Backoff Delay
                                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                                print(f"\n[RATE LIMIT HIT] Model: {model_name}. Waiting for {round(delay, 2)}s (Attempt {attempt + 1}/{MAX_RETRIES})...")
                                time.sleep(delay)
                                
                                # If this was the last attempt, re-raise the error after the sleep
                                if attempt == MAX_RETRIES - 1:
                                    print(f"[FATAL] Failed after {MAX_RETRIES} retries for question: {q}. Skipping.")
                                    result = {"result": "[API_FAILURE] Rate limit exceeded after retries.", "source_documents": []}
                                    break
                            else:
                                # For other exceptions (e.g., 400 Bad Request), log and re-raise immediately
                                print(f"\n[UNEXPECTED ERROR] {e}. Skipping retries for this question.")
                                raise e # Re-raise unexpected errors

                    # If result is still None, it means the API call failed after max retries or hit a non-retryable error
                    if result is None:
                        # Log a placeholder record to mark the failure
                        record = {
                            "Model": model_name,
                            "Chunk_Size": chunk_size,
                            "Retriever_K": retriever_k,
                            "Question": q,
                            "TTFT": None,
                            "Total_Latency": None,
                            "Tokens": 0,
                            "Precision": 0, "Recall": 0, "MRR": 0, "nDCG": 0, "Hit": False,
                            "Timestamp": pd.Timestamp.now()
                        }
                    else:
                        # Continue with metric calculation for successful calls
                        t_end = time.time()

                        ttft = (
                            round(callback.first_token_time - t_start, 4)
                            if callback.first_token_time
                            else None
                        )

                        response = result["result"]
                        docs = result["source_documents"]

                        precision, recall, mrr, ndcg, hit = compute_retrieval_metrics(a, docs)

                        record = {
                            "Model": model_name,
                            "Chunk_Size": chunk_size,
                            "Retriever_K": retriever_k,
                            "Question": q,
                            "TTFT": ttft,
                            "Total_Latency": round(t_end - t_start, 4),
                            "Tokens": len(response.split()),
                            "Precision": precision,
                            "Recall": recall,
                            "MRR": mrr,
                            "nDCG": ndcg,
                            "Hit": hit,
                            "Timestamp": pd.Timestamp.now()
                        }
                    
                    # Write the record (either success or failure)
                    pd.DataFrame([record]).to_csv(
                        OUTPUT_CSV,
                        mode="a",
                        header=write_header,
                        index=False
                    )
                    write_header = False

                print(f"[✓] Completed block for model={model_name}")

    print("\n🏁 Benchmark Finished!")
    return True


# =========================================
# MAIN (Unchanged)
# =========================================
if __name__ == "__main__":
    print("\n🚀 Starting Full Research Benchmark...\n")

    if not os.path.exists(DATA_PATH):
        print(f"FATAL ERROR: File '{DATA_PATH}' not found.")
    else:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            full_dataset = json.load(f)
        
        # --- NEW SAMPLING LOGIC ---
        effective_sample_size = min(BENCHMARK_SAMPLE_SIZE, len(full_dataset))
        dataset_sample = random.sample(full_dataset, effective_sample_size)
        
        print(
            f"Dataset loaded with {len(full_dataset)} entries. "
            f"Benchmarking with a random sample of {effective_sample_size} entries."
        )
        # --------------------------

        run_benchmark(dataset_sample) # Pass the sample to the benchmark function

        print("\n📌 Output saved to:", OUTPUT_CSV)

