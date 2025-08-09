"""LangGraph workflow for policy question answering."""

from typing import List, TypedDict, Annotated
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


class PolicyQueryWorkflow:
    """Handles the LangGraph workflow for policy question answering."""
    
    def __init__(self, api_keys: List[str] = None, model_name: str = "gemini-2.0-flash", model_name_lite: str = "gemini-2.0-flash-lite"):
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
            self.models_lite = [ChatGoogleGenerativeAI(model=model_name_lite)]
        else:
            # Create model instances for each API key
            self.models = []
            self.models_lite = []
            
            for i, api_key in enumerate(api_keys):
                # Temporarily set environment variable for this specific model
                original_key = os.environ.get('GOOGLE_API_KEY')
                os.environ['GOOGLE_API_KEY'] = api_key
                
                try:
                    model = ChatGoogleGenerativeAI(model=model_name)
                    model_lite = ChatGoogleGenerativeAI(model=model_name_lite)
                    self.models.append(model)
                    self.models_lite.append(model_lite)
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
    
    def set_vector_store(self, vector_store: Chroma):
        """Set the vector store for document retrieval."""
        self.vector_store = vector_store
        # Recreate the graphs with the new vector store
        self.graphs = [self._create_graph(i) for i in range(len(self.models))]
    
    def answer_questions(self, questions: List[str]) -> List[str]:
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
                future = executor.submit(self._process_single_question, question, model_idx)
                future_to_index[future] = idx
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    answer = future.result()
                    results[idx] = answer
                except Exception as e:
                    results[idx] = f"Unable to process query due to error: {e}. Please verify the policy documents and try again."
        
        # Return results in original order
        return [results[i] for i in range(len(questions))]
    
    def _process_single_question(self, question: str, model_idx: int) -> str:
        """Process a single question using the specified model index."""
        try:
            # Initialize state with the question
            initial_state = {
                "messages": [HumanMessage(content=question)],
                "refined_query": "",
                "retrieved_context": ""
            }
            
            # Run the workflow with the assigned graph
            final_state = self.graphs[model_idx].invoke(initial_state)
            
            # Extract answer from final message
            if final_state["messages"]:
                last_message = final_state["messages"][-1]
                if isinstance(last_message, AIMessage):
                    return last_message.content or "Unable to generate answer."
                else:
                    return "Unable to generate answer."
            else:
                return "Unable to generate answer."
                
        except Exception as e:
            raise e  # Re-raise to be caught by the caller
    
    def _create_graph(self, model_idx: int):
        """Create the LangGraph workflow."""
        # Create the graph
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("query_refiner", lambda state: self._query_refiner(state, model_idx))
        workflow.add_node("context_retriever", self._context_retriever)
        workflow.add_node("answering_llm", lambda state: self._answering_llm(state, model_idx))
        
        # Define the flow
        workflow.set_entry_point("query_refiner")
        workflow.add_edge("query_refiner", "context_retriever")
        workflow.add_edge("context_retriever", "answering_llm")
        workflow.add_edge("answering_llm", END)
        
        return workflow.compile()
    
    def _query_refiner(self, state: 'State', model_idx: int) -> 'State':
        """Node 1: Refine the user's query for better policy search."""
        messages = state["messages"]
        original_query = messages[-1].content if messages else ""
        
        system_message = SystemMessage(content="""
        You are a query refinement assistant for an insurance policy query assistant.
        Your goal is to refine the given query so it is more precise for the assistant to understand user's intent.
        Refinement rules:
            -Preserve original meaning — do not add, remove, or replace key medical, financial, or policy terms.
            -Clarify, not alter — you may expand abbreviations, fix grammar, or add necessary context from the query itself, but never introduce new procedures, conditions, or assumptions that are not explicitly stated.
            -Enhance search relevance — identify important insurance-related keywords (e.g., coverage, claim eligibility, exclusions, pre-authorization) and incorporate them only if they are directly implied or stated.
            -Remove vague or extraneous language to make the query concise and unambiguous.
        Return only the refined query text, nothing else.
        """)
        
        user_message = HumanMessage(content=f"Original query: {original_query}")
        
        response = self.models_lite[model_idx].invoke([system_message, user_message])
        refined = (response.content or original_query).strip()
        
        return {
            **state,
            "refined_query": refined,
            "messages": state["messages"] + [AIMessage(content=f"Refined query: {refined}")]
        }
    
    def _context_retriever(self, state: 'State') -> 'State':
        """Node 2: Retrieve relevant context from policy documents."""
        refined_query = state.get("refined_query", "")
        
        if self.vector_store is None:
            context = "Vector store not loaded. Please load documents first."
        else:
            try:
                docs = self.vector_store.similarity_search(refined_query, k=5)
                formatted = []
                for i, d in enumerate(docs, 1):
                    src = d.metadata.get("source_url") or d.metadata.get("source_file") or "unknown"
                    formatted.append(f"Clause {i} (Source: {src}):\n{d.page_content}")
                context = "\n\n".join(formatted) if formatted else "No relevant clauses found."
            except Exception as e:
                context = f"Error retrieving clauses: {e}"
        
        return {
            **state,
            "retrieved_context": context
        }
    
    def _answering_llm(self, state: 'State', model_idx: int) -> 'State':
        """Node 3: LLM that answers using the retrieved context."""
        refined_query = state.get("refined_query", "")
        retrieved_context = state.get("retrieved_context", "")
        
        instructions = f"""
        You are an expert AI assistant for insurance policy analysis. Answer the user's question using only the provided context from policy documents.
        
        Context:
        {retrieved_context}
        
        Instructions:
        - Base your answer strictly on the context; no outside facts or assumptions.
        - Do not elaborate; provide only the direct answer citing policy terms and content.
        - Be concise and precise.
        - If the answer is missing, say: "Not specified in the provided policy context."
        """
        
        system_message = SystemMessage(content=instructions)
        user_message = HumanMessage(content=f"Question: {refined_query}")
        
        response = self.models[model_idx].invoke([system_message, user_message])
        
        return {
            **state,
            "messages": state["messages"] + [response]
        }


class State(TypedDict):
    """State definition for the LangGraph workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    refined_query: str
    retrieved_context: str
