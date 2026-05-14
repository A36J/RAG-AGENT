import os
import asyncio
import uuid
import logging
import time
import argparse
import tempfile
import base64
import re
import gc
from dotenv import load_dotenv
from pydantic import SecretStr
import numpy as np

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from llama_parse import LlamaParse

load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)


required_keys = ["OPENROUTER_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME", "LLAMA_CLOUD_API_KEY"]
for key in required_keys:
    if not os.getenv(key):
        raise ValueError(f"{key} is not set in the environment.")

# cleint setup
vision_llm = ChatOpenAI(
    model="qwen/qwen3.5-flash-02-23",  
    api_key=SecretStr(os.getenv("OPENROUTER_API_KEY")),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.0,
    max_retries=1,
    timeout=45.0
)

embeddings = OpenAIEmbeddings(
    model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    api_key=SecretStr(os.getenv("OPENROUTER_API_KEY")),
    base_url="https://openrouter.ai/api/v1",
    model_kwargs={"encoding_format": "float"},              
    check_embedding_ctx_length=False
)

bm25_encoder = BM25Encoder().default()

#  Helpers
# ---------
def l2_normalize(vector: list[float]) -> list[float]:
    vec_array = np.array(vector)
    norm = np.linalg.norm(vec_array)
    if norm == 0: return vector
    return (vec_array / norm).tolist()

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def async_vision_summarize(image_name: str, image_path: str, is_full_page: bool, semaphore: asyncio.Semaphore) -> tuple[str, str]:
    base64_image = encode_image_to_base64(image_path)
    data_uri = f"data:image/jpeg;base64,{base64_image}"
    
    if is_full_page:
        prompt_text = (
            "This is a screenshot of a full document page because the image extractor failed. "
            "IGNORE all standard paragraphs of text. Visually scan the page for charts, graphs, "
            "or diagrams. Describe ONLY the data and concepts in those charts concisely. "
            "If no charts/figures are visible, reply exactly with '[No visual figures detected]'"
        )
    else:
        prompt_text = "Describe this image/chart concisely. Focus on data and key concepts. No filler."

    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": data_uri}}
    ]}]
    
    async with semaphore:
        try:
            response = await asyncio.wait_for(vision_llm.ainvoke(messages), timeout=45.0)
            summary = str(response.content).strip()
            
            if "No visual figures detected" in summary:
                return image_name, ""
                
            return image_name, f"\n\n[Image/Figure Summary: {summary}]\n\n"
        except asyncio.TimeoutError:
            logger.warning(f"Vision API Timeout for {image_name}")
            return image_name, "\n[Image extraction failed: API Timeout]\n"
        except Exception as e:
            logger.warning(f"Vision API Error for {image_name}: {e}")
            return image_name, "\n[Image extraction failed: API Error]\n"

async def async_pinecone_upsert(records: list):
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.IndexAsyncio(os.getenv("PINECONE_INDEX_NAME")) 
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            await index.upsert(vectors=batch)
            logger.info(f"Upserted batch of {len(batch)} vectors to Pinecone.")
    except Exception as e:
        logger.error(f"Failed to upsert to Pinecone: {e}")
        raise e


def extract_with_llamaparse(filepath: str, temp_img_dir: str):
    logger.info(f"--- Linearizing {filepath} with Auto-Mode ---")
    
    
    parser1 = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        
        auto_mode=True,
        auto_mode_trigger_on_image_in_page=True,
        auto_mode_trigger_on_table_in_page=True,
        do_not_unroll_columns=False, 
        continuous_mode=True,
        output_tables_as_HTML=False,
        
        extract_charts=True,
        specialized_image_parsing=True, 
        take_screenshot=False,
        
        user_prompt=(
            "This is a double-column document. Extract text in a single-column, "
            "linear reading order. Do not preserve the side-by-side visual layout. "
            "Extract tables as markdown. Keep image placeholders as ![](filename.png). "
            "No preamble, no extra filler text."
        )
    )
    
    logger.info("Fetching JSON payload from LlamaCloud...")
    json_results = parser1.get_json_result(filepath)
    
    logger.info("Parsing Markdown text into memory-safe page segments...")
    pages_dict = {}
    fallback_idx = 1
    
    for doc in json_results:
        for page in doc.get("pages", []):
            page_num = page.get("page", fallback_idx)
            
            pages_dict[page_num] = page.get("md", "") 
            fallback_idx += 1
            
    if not pages_dict:
        raise ValueError("Markdown extraction resulted in empty text. The document might be empty or unreadable.")

    logger.info("Downloading images to temporary directory...")
    parser2 = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY"))
    raw_images_data = parser2.get_images(json_results, download_path=temp_img_dir)
    
    valid_figures = []
    if raw_images_data:
        for img in raw_images_data:
            img["is_full_page"] = (img.get("type") == "full_page_screenshot")
            valid_figures.append(img)
            
    logger.info(f"Successfully extracted {len(valid_figures)} images to process.")
    return pages_dict, valid_figures


async def process_and_upsert_pipeline(filepath: str, doc_id: str, pages_dict: dict, image_list: list, max_concurrency: int = 5):
    logger.info("--- Starting Async AI Pipeline ---")
    start_time = time.perf_counter()
    llm_semaphore = asyncio.Semaphore(max_concurrency)
    
    try:
        # 1. Process Images into a Lookup Map
        vision_tasks = []
        if image_list:
            logger.info(f"Sending {len(image_list)} images to Vision LLM...")
            for img in image_list:
                img_path = img.get("path")
                is_full_page = img.get("is_full_page", False)
                if img_path and os.path.exists(img_path):
                    img_name = os.path.basename(img_path)
                    vision_tasks.append(async_vision_summarize(img_name, img_path, is_full_page, llm_semaphore))
            
        summaries_results = await asyncio.gather(*vision_tasks) if vision_tasks else []
        
        # building a structured map for tracking usage
        summary_map = {}
        for img_name, summary_text in summaries_results:
            is_fp = next((i.get("is_full_page", False) for i in image_list if i.get("path", "").endswith(img_name)), False)
            summary_map[img_name] = {"summary": summary_text, "used": False, "is_full_page": is_fp}

        # helper to replace placeholder filenames 
        def match_filename(placeholder_path, available_keys):
            base_placeholder = os.path.basename(placeholder_path)
            for key in available_keys:
                if base_placeholder in key or key in base_placeholder:
                    return key
            return None

        # page-by-page extraction
        logger.info("Slicing markdown and chunking by reading order...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", "|", ". ", " "]
        )
        
        final_chunks = []
        global_chunk_idx = 0
        
        for page_num in sorted(pages_dict.keys()):
            page_md = pages_dict[page_num]
            last_idx = 0
            
            
            for match in re.finditer(r'!\[.*?\]\((.*?)\)', page_md):
                start_idx, end_idx = match.span()
                placeholder_filename = match.group(1)
                
                
                text_before = page_md[last_idx:start_idx].strip()
                if text_before:
                    for tc in text_splitter.split_text(text_before):
                        final_chunks.append({
                            "text": tc,
                            "metadata": {"doc_id": doc_id, "type": "text", "page_index": page_num, "chunk_index": global_chunk_idx}
                        })
                        global_chunk_idx += 1
                        
                
                matched_key = match_filename(placeholder_filename, summary_map.keys())
                if matched_key and summary_map[matched_key]["summary"].strip():
                    final_chunks.append({
                        "text": f"Image/Figure Content [{matched_key}]: {summary_map[matched_key]['summary']}",
                        "metadata": {"doc_id": doc_id, "type": "image", "page_index": page_num, "chunk_index": global_chunk_idx}
                    })
                    summary_map[matched_key]["used"] = True
                    global_chunk_idx += 1
                
                last_idx = end_idx
                
            
            text_after = page_md[last_idx:].strip()
            if text_after:
                for tc in text_splitter.split_text(text_after):
                    final_chunks.append({
                        "text": tc,
                        "metadata": {"doc_id": doc_id, "type": "text", "page_index": page_num, "chunk_index": global_chunk_idx}
                    })
                    global_chunk_idx += 1
                    
            
            for k, v in summary_map.items():
                if not v["used"] and v["is_full_page"]:
                    match_page = re.search(r'page_(\d+)', k, re.IGNORECASE)
                    if match_page and int(match_page.group(1)) == page_num:
                        if v["summary"].strip():
                            final_chunks.append({
                                "text": f"Full Page Visual Content [{k}]: {v['summary']}",
                                "metadata": {"doc_id": doc_id, "type": "image", "page_index": page_num, "chunk_index": global_chunk_idx}
                            })
                            global_chunk_idx += 1
                        v["used"] = True

        
        for k, v in summary_map.items():
            if not v["used"] and v["summary"].strip():
                final_chunks.append({
                    "text": f"Unplaced Document Image [{k}]: {v['summary']}",
                    "metadata": {"doc_id": doc_id, "type": "image", "page_index": -1, "chunk_index": global_chunk_idx}
                })
                global_chunk_idx += 1

        
        output_file = f"chunks_inspection_{doc_id}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"--- CHUNK INSPECTION FOR {filepath} ---\n\n")
            for chunk in final_chunks:
                f.write(f"[ID: {chunk['metadata']['chunk_index']} | Page: {chunk['metadata']['page_index']} | Type: {chunk['metadata']['type']}]\n")
                f.write(chunk['text'] + "\n")
                f.write("-" * 50 + "\n\n")
        logger.info(f"💾 Saved {len(final_chunks)} perfectly typed chunks to {output_file}.")

        if not final_chunks:
            logger.warning("No texts to embed after chunking.")
            return

        
        texts = [c["text"] for c in final_chunks]
        
        logger.info(f"Generating dense embeddings for {len(texts)} chunks...")
        raw_dense_vectors = await embeddings.aembed_documents(texts)
        dense_vectors = [l2_normalize(v) for v in raw_dense_vectors]

        logger.info("Generating sparse BM25 vectors...")
        sparse_vectors = await asyncio.to_thread(bm25_encoder.encode_documents, texts)

        # Construct & Upload Payload
        logger.info("Constructing Pinecone payloads...")
        vectors_to_upsert = []
        for i, chunk in enumerate(final_chunks):
            vec_id = f"{doc_id}_{i}"
            
            chunk["metadata"]["text"] = chunk["text"]
            
            vectors_to_upsert.append({
                "id": vec_id,
                "values": dense_vectors[i],
                "sparse_values": sparse_vectors[i],
                "metadata": chunk["metadata"]
            })

        logger.info("Uploading hybrid vectors to Pinecone...")
        await async_pinecone_upsert(vectors_to_upsert)
        
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(f"✅ SUCCESS: Pipeline completed for {filepath} in {processing_time_ms}ms")

    except Exception as e:
        logger.error(f"❌ Pipeline failed for {filepath}: {e}")
        raise e 
    finally:
        gc.collect()

# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a local PDF with LlamaParse and upsert hybrid vectors.")
    parser.add_argument("filename", type=str, help="Name or path of the PDF file in the local directory")

    
    args = parser.parse_args()
    document_id = args.doc_id if args.doc_id else str(uuid.uuid4())
    
    if not os.path.exists(args.filename):
        raise FileNotFoundError(f"File not found at path: {args.filename}")
    
    with tempfile.TemporaryDirectory() as temp_img_dir:
        pages_dict, image_list = extract_with_llamaparse(args.filename, temp_img_dir)
        asyncio.run(process_and_upsert_pipeline(args.filename, document_id, pages_dict, image_list))