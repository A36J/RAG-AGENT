import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.agents import create_agent 
from langchain_core.tools import tool
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

from typing import List, Optional
from pydantic import BaseModel, Field




load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not set in the environment.")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(
    host=os.getenv("PINECONE_INDEX_NAME"),
    pool_threads=50,           
    
)
bm25_encoder = BM25Encoder().default()

class ChunkResult(BaseModel):
    doc_id: Optional[str]
    page: Optional[int]
    chunk_index: Optional[int]
    type: Optional[str]
    text: str

# helper function to apply the alpha scaling
def hybrid_scale(dense, sparse, alpha: float):
    """
    Scales the dense and sparse vectors based on the alpha parameter.
    alpha = 1.0 : Pure Semantic Search
    alpha = 0.0 : Pure Keyword Search
    alpha = 0.5 : Equal Weight Hybrid
    """
   
    alpha = max(0.0, min(1.0, alpha))
    
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

import numpy as np

def l2_normalize(vector: list[float]) -> list[float]:
    """Normalizes a vector to length 1 so Dot Product acts like Cosine Similarity."""
    vec_array = np.array(vector)
    norm = np.linalg.norm(vec_array)
    if norm == 0:
        return vector
    return (vec_array / norm).tolist()


# TOOL SCHEMA
# ------------

class VectorSearchInput(BaseModel):
    query: str = Field(
        ..., 
        description="The text query to search for. Formulate this as a natural language question or a set of specific keywords."
    )
    alpha: float = Field(
        default=0.5, 
        description="Controls the hybrid search balance. Use 1.0 for purely semantic/conceptual meaning. Use 0.0 for exact keyword matches (e.g., specific IDs, acronyms). Use 0.5 for a balanced mix."
    )
   
    top_n: int = Field(
        default=5, 
        description="The number of relevant chunks to retrieve. Max allowed is 8."
    )

    # page filters 
    page_eq: Optional[int] = Field(default=None, description="Filter to an exact page number.")
    page_gt: Optional[int] = Field(default=None, description="Filter to pages strictly greater than this number.")
    page_gte: Optional[int] = Field(default=None, description="Filter to pages greater than or equal to this number.")
    page_lt: Optional[int] = Field(default=None, description="Filter to pages strictly less than this number.")
    page_lte: Optional[int] = Field(default=None, description="Filter to pages less than or equal to this number.")
    page_range: Optional[List[int]] = Field(
        default=None, 
        description="Filter to a range of pages. Pass a list of two integers: [start_page, end_page]."
    )

    # chunk filters
    chunk_eq: Optional[int] = Field(default=None, description="Filter to an exact chunk index.")
    chunk_gt: Optional[int] = Field(default=None, description="Filter to chunk indices strictly greater than this number.")
    chunk_gte: Optional[int] = Field(default=None, description="Filter to chunk indices greater than or equal to this number.")
    chunk_lt: Optional[int] = Field(default=None, description="Filter to chunk indices strictly less than this number.")
    chunk_lte: Optional[int] = Field(default=None, description="Filter to chunk indices less than or equal to this number.")
    chunk_range: Optional[List[int]] = Field(
        default=None, 
        description="Filter to a range of chunk indices. Pass a list of two integers: [start_chunk, end_chunk]."
    )

@tool("vector_search", args_schema=VectorSearchInput)
async def vector_search(

    query: str,
    alpha: float = 0.5, 
    top_n: int = 5,

    page_eq: Optional[int] = None,
    page_gt: Optional[int] = None,
    page_gte: Optional[int] = None,
    page_lt: Optional[int] = None,
    page_lte: Optional[int] = None,
    page_range: Optional[list] = None,

    chunk_eq: Optional[int] = None,
    chunk_gt: Optional[int] = None,
    chunk_gte: Optional[int] = None,
    chunk_lt: Optional[int] = None,
    chunk_lte: Optional[int] = None,
    chunk_range: Optional[list] = None

    
) -> dict[str, list[dict] | str]:
    """
    Performs hybrid (semantic + keyword) search over indexed document chunks using the query and metadata filters.
    The `alpha` parameter controls the search type (0.0 = exact keyword match, 1.0 = pure semantic context, 0.5 = balanced).
    """
    

    warning_msg = ""
    if top_n > 8:
        top_n = 8
        warning_msg += "\n[SYSTEM WARNING: Requested top_n exceeded maximum limit. Results capped at 8.]\n"
        
    if alpha < 0.0 or alpha > 1.0:
            warning_msg += "\n[SYSTEM WARNING: Alpha must be between 0.0 and 1.0. It has been automatically adjusted.]\n"

   
    
    #dense embedding generation
    raw_dense_vector = await embeddings.aembed_query(query) 

    # normalize the query vector 
    dense_vector = l2_normalize(raw_dense_vector)
    
    # generating the sparse BM25 vector
    sparse_vector = bm25_encoder.encode_queries(query)
    
    
    scaled_dense, scaled_sparse = hybrid_scale(dense_vector, sparse_vector, alpha)

   
    metadata_filter = {}

    
    metadata_filter["doc_id"] = "6b66ffa8-67d2-4ed4-96f8-6a80a5f47337"
   

    # page filters
    if page_eq is not None:
        metadata_filter["page_index"] = page_eq
    else:
        page_ops = {}
        if page_gt is not None: page_ops["$gt"] = page_gt
        if page_gte is not None: page_ops["$gte"] = page_gte
        if page_lt is not None: page_ops["$lt"] = page_lt
        if page_lte is not None: page_ops["$lte"] = page_lte
        if page_range:
            page_ops["$gte"] = page_range[0]
            page_ops["$lte"] = page_range[1]
        if page_ops:
            metadata_filter["page"] = page_ops

    # chunk filters
    if chunk_eq is not None:
        metadata_filter["chunk_index"] = chunk_eq
    else:
        chunk_ops = {}
        if chunk_gt is not None: chunk_ops["$gt"] = chunk_gt
        if chunk_gte is not None: chunk_ops["$gte"] = chunk_gte
        if chunk_lt is not None: chunk_ops["$lt"] = chunk_lt
        if chunk_lte is not None: chunk_ops["$lte"] = chunk_lte
        if chunk_range:
            chunk_ops["$gte"] = chunk_range[0]
            chunk_ops["$lte"] = chunk_range[1]
        if chunk_ops:
            metadata_filter["chunk_index"] = chunk_ops

    
    try:
        response = index.query(
            vector=scaled_dense,
            sparse_vector=scaled_sparse,
            top_k=top_n,
            filter=metadata_filter if metadata_filter else None,
            include_metadata=True
        )
    except Exception as e:
        print(f"Failed to retrieve response: {e}")

    print("query end")
    matches = response.get("matches", [])

    if not matches:
        return {
            "results": [],
            "warning": "No relevant chunks found." + warning_msg
        }

  
    structured_chunks = []

    for match in matches:
        meta = match.get("metadata", {})
        structured_chunks.append(
            ChunkResult(
                doc_id=meta.get("doc_id"),
                page=int(meta.get("page")) if meta.get("page") is not None else None,
                chunk_index=int(meta.get("chunk_index")) if meta.get("chunk_index") is not None else None,
                type=meta.get("type"),
                text=meta.get("text", "") 
            )
        )

    return {
        "results": [chunk.dict() for chunk in structured_chunks],
        "warning": warning_msg.strip() if warning_msg else "None"
    }

chat_model = ChatOpenAI(
    model="deepseek/deepseek-v4-flash",
    api_key=SecretStr(api_key),
    base_url="https://openrouter.ai/api/v1",
    streaming=True,
    temperature=0.2,
    default_headers={
        "HTTP-Referer": "http://localhost:5173", 
        "X-Title": "LangGraph Search Agent",
    }
)


embeddings = OpenAIEmbeddings(
    model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    api_key=SecretStr(api_key),
    base_url="https://openrouter.ai/api/v1",
    model_kwargs={"encoding_format": "float"},              
    check_embedding_ctx_length=False
)









systemPrompt="""

You are an elite, precision-focused Document Search & Analysis Agent. Your primary objective is to provide accurate, comprehensive, and highly contextual answers based STRICTLY on the documents provided in the current session. 

Use your vector search tool to effectively find the relevant chunks in the document by making use of the filters and the alpha parameter to pinpoint the exact chunk.
If repeated (more than 8 searches) use doesnt retrieve chunks relevant to the user query state you couldnt retieve the answer .
"""
tools = [vector_search]

def build_agent(checkpointer):
    return create_agent(
        model=chat_model,
        system_prompt=systemPrompt, 
        tools=tools,
        checkpointer=checkpointer
    )