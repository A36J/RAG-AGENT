# STRATEGY.md

Here is a look under the hood at how this RAG app was built, the roadblocks I hit along the way, and why I made the architectural choices I did.

## How the Pipeline Works

The app runs on a Streamlit frontend, but the backend process is broken into two main phases:

**1. Preparing the Data (Ingestion):**
* I take the raw PDF and pass it to LlamaParse to extract the text, keeping the markdown formatting and tables perfectly intact.
* LlamaParse also isolates the charts and figures into physical image files. I send those images to Gemini 3.1 Flash to generate descriptive summaries.
* I chunk the text and the image summaries, convert them into both dense and sparse vectors, and upload everything to Pinecone. 

**2. Answering Questions (Retrieval):**
* When a user asks a question, an AI agent takes over. Instead of blindly searching the database with the user's exact words, the agent uses a "search tool." 
* It figures out the best search terms, queries Pinecone, and evaluates the chunks it gets back. If the info isn't good enough, the agent will rewrite its search query and try again before finally streaming an answer to the user.

---

## 1. The Discovery & Fix Log

**The PDF Parsing Nightmare**
My biggest headache was just getting clean text out of the PDF. The document is a complex academic paper with two-column layouts and lots of math equations. 
* First, I tried OpenDataLoader, but it struggled with the layout. It kept reading text from *inside* the charts and mixing it with the paragraph text.
* Next, I tried Docling, but it was incredibly heavy to run locally and froze up my machine. 
* I moved to the Unstructured API, which was better, but it occasionally fragmented the data tables.
* **The Fix:** I finally landed on LlamaParse Premium. It flawlessly preserved the markdown tables and pulled the charts out as clean, separate image files that I could process with a Vision model.

**The "Event Loop" Crashes**
Once I got LlamaParse working, my Python script started crashing with an "Event loop is closed" error. This happened because I was trying to mix standard, synchronous code (LlamaParse extracting files) with high-speed asynchronous code (uploading chunks to Pinecone and calling Gemini).
* **The Fix:** I completely separated the two steps. I now run the LlamaParse extraction as a standard, isolated step that saves the text and images to a temporary folder. Once that finishes and closes, the async AI pipeline spins up to handle the embedding and uploading. 

---

## 2. Design Decisions

**Why an Agent instead of standard RAG?**
Standard RAG pipelines just take the user's prompt and search the database. If the user asks a poorly worded question, they get bad search results. By giving my AI a search tool, the agent acts like a human researcher. It thinks about what the user actually wants, formulates a smart search query, checks the results, and will search again if it didn't find the right answer on the first try.

**Hybrid Search over Standard Vectors**
For technical documents, standard semantic search (dense vectors) isn't enough. If a user searches for a specific acronym or product name, you need exact keyword matching. I used a hybrid approach: dense vectors handle the conceptual questions, while sparse vectors (BM25) handle the exact keyword matches. 

**Skipping the Reranker**
Usually, you want a reranker (like Cohere) to sort your retrieved chunks. However, since I am only searching a single document, and my agent is smart enough to re-evaluate and search again if it gets bad context, adding a reranker felt like unnecessary overhead that would just slow the app down.

---

## 3. Quality Assurance (Production Strategy)

To systematically measure how well this pipeline actually works in a real-world setting, I would use an "LLM-as-a-Judge" framework to test three core metrics:

1. **Context Relevance:** Did the search tool actually grab useful information, or did it flood the context window with irrelevant noise?
2. **Faithfulness:** Is the final answer strictly based on the text retrieved from the PDF, or did the AI hallucinate and use its own outside knowledge?
3. **Answer Relevance:** Did the final response actually answer the user's original question? 
