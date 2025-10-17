import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import textwrap
import tempfile

# ------------------------------------------------------------
# STEP 1: Extract text from PDFs
# ------------------------------------------------------------
def extract_text_from_pdfs(uploaded_files):
    text = ""
    for file in uploaded_files:
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text("text") + "\n"
    return text


# ------------------------------------------------------------
# STEP 2: Chunk text
# ------------------------------------------------------------
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ------------------------------------------------------------
# STEP 3: Build FAISS index
# ------------------------------------------------------------
def build_faiss_index(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, model, chunks


# ------------------------------------------------------------
# STEP 4: Initialize LLM (IBM Granite 3.2 2B Instruct)
# ------------------------------------------------------------
def load_llm():
    model_name = "ibm-granite/granite-3.2-2b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)
    return pipe


# ------------------------------------------------------------
# STEP 5: Process PDFs (end-to-end preprocessing)
# ------------------------------------------------------------
def process_pdfs(uploaded_files):
    full_text = extract_text_from_pdfs(uploaded_files)
    chunks = chunk_text(full_text)
    index, emb_model, all_chunks = build_faiss_index(chunks)
    llm_pipe = load_llm()

    return {
        "index": index,
        "embedding_model": emb_model,
        "chunks": all_chunks,
        "llm_pipe": llm_pipe,
    }


# ------------------------------------------------------------
# STEP 6: Semantic Retrieval
# ------------------------------------------------------------
def retrieve_chunks(query, context_data, top_k=3):
    model = context_data["embedding_model"]
    index = context_data["index"]
    chunks = context_data["chunks"]

    query_emb = model.encode([query])
    distances, indices = index.search(np.array(query_emb), top_k)
    selected = [chunks[i] for i in indices[0]]
    return selected


# ------------------------------------------------------------
# STEP 7: Generate Answer
# ------------------------------------------------------------
def generate_answer(query, context, llm_pipe):
    prompt = f"""
You are StudyMate, an intelligent academic assistant.
Answer the question using the context below. 
If the answer isn't in the context, clearly say "I couldn’t find that information in the uploaded documents."

Question: {query}
Context:
{context}

Answer:
"""
    response = llm_pipe(prompt)[0]["generated_text"]
    answer = response.split("Answer:")[-1].strip()
    return textwrap.fill(answer, width=90)


# ------------------------------------------------------------
# STEP 8: Answer a User Query
# ------------------------------------------------------------
def answer_question(query, context_data, top_k=3):
    retrieved_chunks = retrieve_chunks(query, context_data, top_k)
    context = "\n\n".join(retrieved_chunks)
    answer = generate_answer(query, context, context_data["llm_pipe"])
    return answer, context

