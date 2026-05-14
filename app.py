import streamlit as st
import uuid
import json
import asyncio
from langchain_core.messages import AIMessage, ToolMessage,HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig


from api.agent import build_agent

st.set_page_config(page_title="Single-Turn RAG Agent", layout="centered")


@st.cache_resource
def get_agent():

    checkpointer = MemorySaver()
    return build_agent(checkpointer)

agent = get_agent()

st.title("Single-Turn RAG Agent")
st.markdown("Ask a query. The agent will autonomously route, search, and compile an answer.")

# session state to ensure UI persistence 
if "history" not in st.session_state:
    st.session_state.history = []

# displaying past turns strictly for UI 
for interaction in st.session_state.history:
    with st.chat_message("user"):
        st.write(interaction["user"])
    with st.chat_message("assistant"):
        for i, chunks in enumerate(interaction["tool_chunks"]):
            with st.expander(f"🛠️ Search Iteration {i+1}", expanded=False):
                st.write(chunks)
        st.markdown(interaction["final_answer"])


user_query = st.chat_input("Enter your query here...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning and searching the database..."):
            
            # generate a unique thread ID for this single-turn query
            thread_id = str(uuid.uuid4())
            config:RunnableConfig = {"configurable": {"thread_id": thread_id}}
            
           
            initial_state = {"messages": [HumanMessage(content=user_query)]}
            final_state = asyncio.run(agent.ainvoke(initial_state, config=config))
            
            messages = final_state.get("messages", [])
            
            tool_call_counter = 1
            final_answer = ""
            run_chunks = []
            
            
            for msg in messages:
                
                # Check for Tool Messages (Retrieved chunks from vector_search)
                if isinstance(msg, ToolMessage):
                    with st.expander(f"🛠️ Search Iteration {tool_call_counter}", expanded=False):
                        try:
                            
                            parsed_chunks = json.loads(msg.content)
                            st.json(parsed_chunks)
                            run_chunks.append(parsed_chunks)
                        except json.JSONDecodeError:
                            
                            st.write(msg.content)
                            run_chunks.append(msg.content)
                    tool_call_counter += 1
                    
                
                elif isinstance(msg, AIMessage):
                   
                    if msg.content and not msg.tool_calls:
                        final_answer = msg.content
                        st.markdown(final_answer)
            
            # Save to UI history so it survives Streamlit re-renders
            st.session_state.history.append({
                "user": user_query,
                "tool_chunks": run_chunks,
                "final_answer": final_answer
            })