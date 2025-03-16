from typing import List, Tuple, Any, Dict
import os
import re
from dotenv import load_dotenv
from agent_graph.build_full_graph import AgentGraph
from utils.app_utils import load_config, get_available_collections, get_available_databases


class ChatBot:
    """Chatbot interface for the agent system."""
    
    _instance = None
    _is_initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatBot, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ChatBot._is_initialized:
            load_dotenv()
            
            # Load configuration
            self.config = load_config()
            
            # Initialize agent
            self.agent = self._initialize_agent()
            
            # Track error counts to prevent repeated errors
            self.error_counts = {}
            
            ChatBot._is_initialized = True
    
    def _initialize_agent(self) -> AgentGraph:
        """Initialize the agent with all available tools."""
        agent = AgentGraph(config_path="configs/custom_config.yml")
        
        # Add RAG tools for each collection
        collections = get_available_collections()
        for collection in collections:
            agent.add_rag_tool(
                collection_name=collection,
                tool_name=f"query_{collection}",
                description=f"Use this tool to ask questions about documents in the {collection} collection."
            )
        
        # Add SQL tools for each database
        databases = get_available_databases()
        for db_type, db_names in databases.items():
            for db_name in db_names:
                agent.add_sql_tool(
                    db_name=db_name,
                    tool_name=f"query_{db_name}",
                    description=f"Use this tool to query the {db_name} {db_type} database."
                )
        
        # Build the agent
        agent.build()
        
        return agent
    
    @classmethod
    def respond(cls, history: List[Tuple[str, str]], message: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Process a message and return a response."""
        if not message:
            return "", history

        if cls._instance is None:
            cls._instance = ChatBot()

        try:
            # Preprocess the query
            processed_message = cls._instance._preprocess_query(message)
            
            # Get response from agent
            response = cls._instance.agent.run(processed_message)
            
            # Update history
            history.append((message, response))
            
            return "", history
            
        except Exception as e:
            error_message = cls._instance._format_error_message(e, message)
            history.append((message, error_message))
            return "", history

    def _preprocess_query(self, message: str) -> str:
        """Preprocess the user's query to handle special cases."""
        if any(keyword in message.lower() for keyword in ["flight", "airport", "travel", "airline"]):
            airport_codes = re.findall(r'\b[A-Z]{3}\b', message)
            if airport_codes:
                return (
                    f"{message}\n\n"
                    f"Note: This query involves airport codes: {', '.join(airport_codes)}. "
                    f"When working with these codes, ensure they are treated as text values, not numbers."
                )
        return message

    def _format_error_message(self, error: Exception, message: str) -> str:
        """Format error message for user display."""
        error_type = type(error).__name__
        error_msg = str(error)
        
        if "must be real number" in error_msg:
            return (
                "Error: I encountered an issue with numeric conversion. This often happens with codes "
                "like airport codes (e.g., CDG, LHR) that should be treated as text.\n\n"
                "Please try again with a more specific query."
            )
        elif "parsing errors" in error_msg.lower() or "parsing llm output" in error_msg.lower():
            return (
                "I encountered an issue understanding how to process your query. This typically happens when:\n\n"
                "1. The query is ambiguous or could be interpreted in multiple ways\n"
                "2. The query requires accessing multiple data sources simultaneously\n"
                "3. The query format doesn't match the expected database structure\n\n"
                "Please try rephrasing your question to be more specific about what you're looking for."
            )
        else:
            return f"Error processing query: {error_msg}"
