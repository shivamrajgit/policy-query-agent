"""LangGraph workflow for policy question answering."""

import time
import logging
from typing import List, TypedDict, Annotated
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


class PolicyQueryWorkflow:
    """Handles the LangGraph workflow for policy question answering."""
    
    def __init__(self, api_keys: List[str] = None, model_name: str = "gemini-2.5-flash"):
        self.logger = logging.getLogger(__name__)
        # Load environment variables
        load_dotenv()
        
        # If no API keys provided, try to load from environment
        if api_keys is None:
            api_keys_str = os.getenv('GOOGLE_API_KEYS')
            if api_keys_str:
                try:
                    parsed_keys = json.loads(api_keys_str)
                    api_keys = [key for key in parsed_keys if key and key.strip()]
                    if api_keys:
                        print(f"Loaded {len(api_keys)} API keys from environment")
                    else:
                        print("No valid API keys found in GOOGLE_API_KEYS")
                        api_keys = None
                except json.JSONDecodeError as e:
                    print(f"Error parsing GOOGLE_API_KEYS: {e}")
                    api_keys = None
            else:
                print("GOOGLE_API_KEYS not found in environment")
        
        if api_keys is None:
            # Fallback to single model if no API keys provided
            self.models = [ChatGoogleGenerativeAI(model=model_name)]
        else:
            # Create model instances for each API key
            self.models = []
            
            for i, api_key in enumerate(api_keys):
                # Temporarily set environment variable for this specific model
                original_key = os.environ.get('GOOGLE_API_KEY')
                os.environ['GOOGLE_API_KEY'] = api_key
                
                try:
                    model = ChatGoogleGenerativeAI(model=model_name)
                    self.models.append(model)
                    print(f"Created model {i+1} with API key: {api_key[:10]}...{api_key[-4:]}")
                finally:
                    # Restore original key or remove if it wasn't set
                    if original_key:
                        os.environ['GOOGLE_API_KEY'] = original_key
                    else:
                        os.environ.pop('GOOGLE_API_KEY', None)
        
        self.api_keys = api_keys or ["default"]
        self.vector_store = None
        self.graphs = None
    
    def set_vector_store(self, vector_store: FAISS):
        """Set the vector store for document retrieval."""
        self.vector_store = vector_store
        # Recreate the graphs with the new vector store
        self.graphs = [self._create_graph(i) for i in range(len(self.models))]
    
    def answer_questions(self, questions: List[str], request_id: str = "unknown") -> List[str]:
        """Answer multiple questions using parallel processing with multiple API keys."""
        if self.vector_store is None:
            raise ValueError("Vector store not set. Please call set_vector_store() first.")
        if not questions:
            return []
        
        if self.graphs is None:
            self.graphs = [self._create_graph(i) for i in range(len(self.models))]
        
        # Use ThreadPoolExecutor to process questions in parallel
        max_workers = len(self.models)
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all questions to the thread pool
            future_to_index = {}
            
            for idx, question in enumerate(questions):
                # Assign API key/model index based on round-robin
                model_idx = idx % len(self.models)
                future = executor.submit(self._process_single_question, question, model_idx, idx, request_id)
                future_to_index[future] = idx
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                completed_count += 1
                try:
                    answer = future.result()
                    results[idx] = answer
                    print(f"  Q{idx+1} completed ({completed_count}/{len(questions)})")
                except Exception as e:
                    error_msg = f"Unable to process query due to error: {e}. Please verify the policy documents and try again."
                    results[idx] = error_msg
                    print(f"  Q{idx+1} failed ({completed_count}/{len(questions)}): {e}")
        
        # Return results in original order
        return [results[i] for i in range(len(questions))]
    
    def _process_single_question(self, question: str, model_idx: int, question_idx: int, request_id: str) -> str:
        """Process a single question using the specified model index."""
        start_time = time.perf_counter()
        
        try:            
            # Initialize state with the question
            initial_state = {
                "messages": [HumanMessage(content=question)],
                "retrieved_text": ""
            }
            
            # Run the workflow with the assigned graph
            workflow_start = time.perf_counter()
            final_state = self.graphs[model_idx].invoke(initial_state)
            workflow_time = time.perf_counter() - workflow_start
            
            # Extract answer from final message
            if final_state["messages"]:
                last_message = final_state["messages"][-1]
                if isinstance(last_message, AIMessage):
                    answer = last_message.content or "Unable to generate answer."
                else:
                    answer = "Unable to generate answer."
            else:
                answer = "Unable to generate answer."
            
            # Record timing
            total_time = time.perf_counter() - start_time
            
            return answer
                
        except Exception as e:
            total_time = time.perf_counter() - start_time
            raise e  # Re-raise to be caught by the caller
    
    def _create_graph(self, model_idx: int):
        """Create the LangGraph workflow."""
        # Create the graph
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("context_retriever", self._context_retriever)
        workflow.add_node("answering_llm", lambda state: self._answering_llm(state, model_idx))
        
        # Define the flow
        workflow.set_entry_point("context_retriever")
        workflow.add_edge("context_retriever", "answering_llm")
        workflow.add_edge("answering_llm", END)
        
        return workflow.compile()
    
    def _context_retriever(self, state: 'State') -> 'State':
        """Node 1: Retrieve relevant context from policy documents using the original query."""
        start_time = time.perf_counter()
        messages = state["messages"]
        original_query = messages[-1].content if messages else ""
        
        if self.vector_store is None:
            retrieved_text = "Vector store not loaded. Please load documents first."
        else:
            try:
                self.logger.debug(f"[CONTEXT] Retrieving context for: {original_query[:50]}...")
                docs = self.vector_store.similarity_search(original_query, k=4)
                formatted = []
                for i, d in enumerate(docs, 1):
                    src = d.metadata.get("source_url") or d.metadata.get("source_file") or "unknown"
                    formatted.append(f"Clause {i} (Source: {src}):\n{d.page_content}")
                retrieved_text = "\n\n".join(formatted) if formatted else "No relevant clauses found."
                
                elapsed = time.perf_counter() - start_time
                self.logger.debug(f"[CONTEXT_DONE] Context retrieval completed in {elapsed:.4f}s, found {len(docs)} relevant clauses")
            except Exception as e:
                retrieved_text = f"Error retrieving clauses: {e}"
                self.logger.error(f"[CONTEXT_ERROR] Context retrieval failed: {e}")
        
        return {
            **state,
            "retrieved_text": retrieved_text
        }
    
    def _answering_llm(self, state: 'State', model_idx: int) -> 'State':
        """Node 2: LLM that answers using the retrieved context."""
        start_time = time.perf_counter()
        original_query = state["messages"][0].content if state["messages"] else ""
        retrieved_text = state.get("retrieved_text", "")
        
        self.logger.debug(f"[LLM] Generating answer with Model-{model_idx}")
        
        instructions = f"""
        You are an expert AI assistant for insurance policy analysis.
        
        Context:
        {retrieved_text}
        
        Instructions:
        - Only answer using facts from provided context, no outside facts or assumptions.
        - Do not elaborate or add, keep it consise and crisp (1-2 sentences)
        - If the answer is missing, say: "Not specified in the provided policy context."
        """
        
        system_message = SystemMessage(content=instructions)
        user_message = HumanMessage(content=f"Question: {original_query}")
        
        llm_start = time.perf_counter()
        response = self.models[model_idx].invoke([system_message, user_message])
        llm_time = time.perf_counter() - llm_start
        
        total_time = time.perf_counter() - start_time
        self.logger.debug(f"[LLM_DONE] LLM response generated in {llm_time:.4f}s (total step: {total_time:.4f}s)")
        
        return {
            **state,
            "messages": state["messages"] + [response]
        }


class State(TypedDict):
    """State definition for the LangGraph workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    retrieved_text: str
