import streamlit as st
import os
import json
import pandas as pd
from dotenv import load_dotenv

# LangChain Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document

# Load environment variables (reads GROQ_API_KEY from .env file)
load_dotenv()

# ============================================================
# 1️⃣ CONFIGURATION
# ============================================================
DATA_PATH = "IndicLegalQA Dataset_10K.json"
CHROMA_DIR = "./chroma_store_legal"
CSV_OUTPUT_PATH = "rag_batch_results.csv"

# Retrieval & LLM Settings
GROQ_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
RETRIEVER_K = 4

# ============================================================
# 2️⃣ STRUCTURED LEGAL PROMPT TEMPLATE
# ============================================================
LEGAL_PROMPT_TEMPLATE = """
You are a highly analytical and objective Legal Assistant. Your task is to provide
a structured, fact-based response ONLY using the provided legal context.
Do not introduce external information or speculation.

Context: {context}
Question: {question}

Follow these steps for your final answer:
1. Legal Issue: Identify the central legal question or concept.
2. Relevant Authority: Cite the specific section, article, or case summary from the 'Context' that applies.
3. Reasoning: Analyze how the cited authority applies to the question.
4. Conclusion: Provide a final, concise, and definitive legal summary.
"""

LEGAL_PROMPT = PromptTemplate(template=LEGAL_PROMPT_TEMPLATE,
                              input_variables=["question", "context"])

# ============================================================
# 3️⃣ CACHED FUNCTIONS
# ============================================================
@st.cache_resource(show_spinner=False)
def setup_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(CHROMA_DIR):
        vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        st.info(f"Loaded existing vector store from {CHROMA_DIR}")
        return vectordb

    st.info("Creating new embeddings...")

    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            legal_docs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.error(f"Could not load or parse data file at {DATA_PATH}.")
        return None

    if not isinstance(legal_docs, list) or not legal_docs:
        st.error(f"Data file {DATA_PATH} is empty or invalid.")
        return None

    documents = []
    for i, doc in enumerate(legal_docs):
        content = doc.get("content") or doc.get("text")
        source = doc.get("source", f"document_{i+1}")
        if content and isinstance(content, str) and content.strip():
            documents.append(Document(page_content=content,
                                      metadata={"source": source}))

    if not documents:
        st.error("No valid documents found.")
        return None

    st.info(f"Loaded {len(documents)} documents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    texts = splitter.split_documents(documents)

    if not texts:
        st.error("Text splitting resulted in zero chunks.")
        return None

    st.info(f"Split into {len(texts)} chunks.")

    vectordb = Chroma.from_documents(texts,
                                     embedding=embeddings,
                                     persist_directory=CHROMA_DIR)
    vectordb.persist()
    st.success(f"Vector store saved in {CHROMA_DIR}")

    return vectordb


@st.cache_resource(show_spinner=False)
def initialize_rag_chain(_vectordb):
    llm = ChatGroq(
        model_name=GROQ_MODEL,
        temperature=0.1,
        api_key=os.environ.get("GROQ_API_KEY")
    )

    retriever = _vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "lambda_mult": 0.7}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": LEGAL_PROMPT},
        return_source_documents=True
    )

    return qa_chain

# ============================================================
# 4️⃣ STREAMLIT USER INTERFACE
# ============================================================
def main_app():
    st.set_page_config(page_title="Legal RAG Assistant", layout="wide")
    st.title("⚖️ Intelligent Legal Document Assistant")
    st.subheader(f"Using LangChain + Groq ({GROQ_MODEL})")
    st.markdown("---")

    if not os.environ.get("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY in .env")
        return

    with st.spinner("Loading resources..."):
        vectordb = setup_vector_store()
        if vectordb is None:
            return
        qa_chain = initialize_rag_chain(vectordb)

    st.markdown("---")
    st.markdown("## Batch Query Processing")

    batch_query_input = st.text_area(
        "Enter queries (one per line):",
        height=200
    )

    current_batch_results = []

    if st.button("Run Batch"):
        queries_to_run = [q.strip() for q in batch_query_input.splitlines() if q.strip()]

        if not queries_to_run:
            st.warning("Please enter queries.")
            return

        st.success(f"Running {len(queries_to_run)} queries...")
        results_placeholder = st.container()
        write_header = not os.path.exists(CSV_OUTPUT_PATH)

        for idx, query in enumerate(queries_to_run):
            current_result_row = {
                "Timestamp": pd.Timestamp.now(),
                "Query_ID_Batch": idx+1,
                "Query": query,
                "Response": "",
                "Source_Count": 0,
                "Sources_List": "",
                "Status": "Failed"
            }

            with results_placeholder:
                st.markdown(f"### Q{idx+1}: {query}")

                with st.spinner(f"Processing Q{idx+1}..."):
                    try:
                        result = qa_chain.invoke({"query": query})
                        response_text = result['result']
                        source_docs = result.get('source_documents', [])

                        st.markdown(response_text)

                        sources_list = [doc.metadata.get('source', 'N/A') for doc in source_docs]

                        current_result_row["Response"] = response_text.replace('\n', ' ').replace('\r', '')
                        current_result_row["Source_Count"] = len(sources_list)
                        current_result_row["Sources_List"] = "; ".join(sources_list)
                        current_result_row["Status"] = "Success"

                        with st.expander(f"Sources ({len(source_docs)})"):
                            if source_docs:
                                for i, doc in enumerate(source_docs):
                                    st.caption(f"Source {i+1}: {doc.metadata.get('source', 'N/A')}")
                                    st.code(doc.page_content, language="markdown")
                                    st.markdown("---")
                            else:
                                st.warning("No documents retrieved")

                    except Exception as e:
                        st.error(f"Error: {e}")
                        current_result_row["Response"] = f"ERROR: {e}"
                        current_result_row["Status"] = "Error"

            df_row = pd.DataFrame([current_result_row])
            df_row.to_csv(CSV_OUTPUT_PATH, mode='a', header=write_header, index=False)
            write_header = False
            current_batch_results.append(current_result_row)

        if current_batch_results:
            st.markdown("---")
            st.success("Batch completed & saved to CSV")

            try:
                full_df = pd.read_csv(CSV_OUTPUT_PATH)
                csv_file = full_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download Full Results CSV",
                                   data=csv_file,
                                   file_name=CSV_OUTPUT_PATH,
                                   mime='text/csv')
            except Exception as e:
                st.warning(f"Download failed: {e}")

        st.balloons()


if __name__ == "__main__":
    main_app()
