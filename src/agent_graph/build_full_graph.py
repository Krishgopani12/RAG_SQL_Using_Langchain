import os
from typing import Dict, List, Optional, Any
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
import yaml
from pyprojroot import here
from dotenv import load_dotenv
import re

from .generic_rag_tool import create_rag_tool_from_config, GenericRAGTool
from .generic_sql_agent import create_sql_agent_from_config, SQLDatabaseAgent


class AgentCallback(BaseCallbackHandler):
    """Callback handler for agent execution."""
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when agent takes an action."""
        print(f"\nAgent action: {action.tool}\nAction input: {action.tool_input}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when agent finishes execution."""
        print(f"\nAgent finished: {finish.return_values}")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM completes a generation."""
        pass


class AgentGraph:
    """A flexible agent graph that can work with any combination of tools."""

    def __init__(self, config_path: str):
        """
        Initialize the agent graph with configuration.
        
        Args:
            config_path (str): Path to the configuration file
        """
        with open(here(config_path)) as f:
            self.config = yaml.safe_load(f)
        
        self.llm_config = self.config['llm_config']
        self.tools: List[BaseTool] = []
        self.agent_executor = None

    def add_rag_tool(self, collection_name: str, tool_name: str, description: str) -> None:
        """
        Add a RAG tool to the agent.
        
        Args:
            collection_name (str): Name of the document collection
            tool_name (str): Name for the tool
            description (str): Description of what the tool does
        """
        rag_tool = create_rag_tool_from_config(
            config_path="configs/custom_config.yml",
            collection_name=collection_name
        )
        
        # Enhanced description with specific collection context
        enhanced_description = (
            f"{description} This tool specifically searches the '{collection_name}' collection. "
            f"Use this when you need information about {collection_name.replace('_', ' ')}. "
            f"For best results, provide specific questions related to this collection's content."
        )
        
        @tool
        def rag_query(query: str, run_manager=None) -> str:
            """
            {enhanced_description}
            
            Args:
                query (str): The question to ask about the documents
                run_manager: Optional run manager for callbacks
            
            Returns:
                str: Answer from the document collection
            """
            try:
                result = rag_tool.query(query)
                return f"From {collection_name}: {result['answer']}"
            except Exception as e:
                return f"Error querying {collection_name}: {str(e)}"
        
        # Set the tool's name after creation
        rag_query.name = tool_name
        self.tools.append(rag_query)

    def add_sql_tool(self, db_name: str, tool_name: str, description: str) -> None:
        """
        Add a SQL tool to the agent.
        
        Args:
            db_name (str): Name of the database configuration
            tool_name (str): Name for the tool
            description (str): Description of what the tool does
        """
        try:
            sql_agent = create_sql_agent_from_config(
                config_path="configs/custom_config.yml",
                db_name=db_name
            )
            
            # Get database schema information safely
            try:
                table_info = sql_agent.get_table_info()
            except ValueError as ve:
                if "Environment variable" in str(ve):
                    error_msg = (
                        f"Database '{db_name}' requires environment variables to be set. "
                        f"Error: {str(ve)}"
                    )
                    print(f"Warning: {error_msg}")
                    table_info = "Schema information unavailable - missing environment variables"
                else:
                    table_info = f"Error getting detailed table info: {str(ve)}"
            except Exception as e:
                table_info = f"Error getting detailed table info: {str(e)}"
                # Fallback to just table names
                try:
                    table_names = sql_agent.get_table_names()
                    table_info = f"Available tables: {', '.join(table_names)}"
                except Exception:
                    table_info = "Schema information unavailable"
            
            # Enhanced description with database schema context
            enhanced_description = (
                f"{description} This tool queries the '{db_name}' database. "
                f"Database structure: {table_info}. "
                f"Use this when you need structured data from {db_name}."
            )
            
            @tool
            def sql_query(query: str, run_manager=None) -> str:
                """
                {enhanced_description}
                
                Args:
                    query (str): The natural language query to run against the database
                    run_manager: Optional run manager for callbacks

    Returns:
                    str: Results from the database query
                """
                try:
                    result = sql_agent.run_query(query)
                    return f"From {db_name}: {result}"
                except ValueError as ve:
                    if "Environment variable" in str(ve):
                        return f"Error: Database '{db_name}' is not accessible - missing environment variables. Please check your configuration."
                    return f"Error querying {db_name}: {str(ve)}"
                except Exception as e:
                    return f"Error querying {db_name}: {str(e)}"
            
            # Set the tool's name after creation
            sql_query.name = tool_name
            self.tools.append(sql_query)
            
        except ValueError as ve:
            if "Environment variable" in str(ve):
                error_msg = f"Database '{db_name}' requires environment variables to be set: {str(ve)}"
            else:
                error_msg = str(ve)
            print(f"Warning: Failed to create SQL tool for {db_name}: {error_msg}")
            
            @tool
            def sql_query(query: str, run_manager=None) -> str:
                """This database is currently unavailable due to configuration issues."""
                return f"Error: The {db_name} database is not available: {error_msg}"
            
            sql_query.name = tool_name
            self.tools.append(sql_query)
        except Exception as e:
            print(f"Warning: Failed to create SQL tool for {db_name}: {str(e)}")
            
            @tool
            def sql_query(query: str, run_manager=None) -> str:
                """This database is currently unavailable."""
                return f"Error: The {db_name} database is not available: {str(e)}"
            
            sql_query.name = tool_name
            self.tools.append(sql_query)

    def build(self) -> None:
        """Build the agent with all added tools."""
        if not self.tools:
            raise ValueError("No tools have been added to the agent")

        # Create LLM
        llm = ChatOpenAI(
            model=self.llm_config.get('model', 'gpt-4'),
            temperature=self.llm_config.get('temperature', 0.0)
        )

        # Create agent prompt with enhanced reasoning
        prompt = PromptTemplate.from_template(
            """You are a helpful AI assistant that can use various tools to answer questions.
            You have access to multiple data sources through these tools:
            
            {tools}
            
            Important Guidelines:
            1. For questions about airline policies, use the airline-policies collection
            2. For questions about stories, use the stories collection
            3. For travel-related queries, use the travel-db database
            4. For music and media queries, use the chinook-db database
            5. When a question requires multiple sources, use relevant tools in sequence
            6. Always explain which sources you're using and why
            
            Use the following format:
            
            Question: the input question you must answer
            Thought: carefully consider which data sources are needed
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer with clear attribution to sources
            
            Begin!
            
            Question: {input}
            Thought: {agent_scratchpad}"""
        )

        # Create agent
        agent = create_react_agent(llm, self.tools, prompt)
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            early_stopping_method="generate"
        )

    def run(self, query: str) -> str:
        """
        Run a query through the agent.
        
        Args:
            query (str): The query to process
            
        Returns:
            str: Response from the agent
        """
        if not self.agent_executor:
            raise ValueError("Agent has not been built. Call build() first.")
        
        try:
            # Pre-process travel-related queries
            if any(keyword in query.lower() for keyword in ["flight", "airport", "travel"]):
                airport_codes = re.findall(r'\b[A-Z]{3}\b', query)
                if airport_codes:
                    # Add guidance for the agent
                    query = (
                        f"{query}\n\n"
                        f"Note: This query involves airport codes: {', '.join(airport_codes)}. "
                        f"When working with these codes, ensure they are treated as text values, not numbers."
                    )
            
            # Create a callback handler for tracking
            callback = AgentCallback()
            
            # Use invoke instead of run to address deprecation warning
            try:
                response = self.agent_executor.invoke(
                    {"input": query},
                    callbacks=[callback],
                    handle_parsing_errors=True
                )
                
                # Extract the output from the response
                if isinstance(response, dict) and "output" in response:
                    return response["output"]
                elif isinstance(response, str):
                    return response
                else:
                    return f"Received response in unexpected format: {str(response)}"
            except AttributeError:
                # Fallback to direct run method if invoke is not available
                print("Warning: Using deprecated run method as fallback")
                return self.agent_executor.run(input=query)
                
        except ValueError as e:
            error_msg = str(e)
            if "parsing errors" in error_msg.lower() or "parsing llm output" in error_msg.lower():
                return (
                    f"I encountered an issue understanding how to process your query. This typically happens when:\n\n"
                    f"1. The query is ambiguous or could be interpreted in multiple ways\n"
                    f"2. The query requires accessing multiple data sources simultaneously\n"
                    f"3. The query format doesn't match the expected database structure\n\n"
                    f"Please try rephrasing your question to be more specific about what you're looking for."
                )
            elif "must be real number" in error_msg:
                return (
                    f"Error: The query involves data that couldn't be properly converted to numeric format.\n\n"
                    f"This often happens when working with codes (like airport codes) that look like numbers "
                    f"but should be treated as text. Please try again with a more specific query that clarifies "
                    f"the data types, or use standard codes where applicable."
                )
            else:
                return f"Error: {error_msg}"
        except Exception as e:
            return f"Error: {str(e)}"


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Create agent graph
    agent = AgentGraph(config_path="configs/custom_config.yml")
    
    # Add tools
    agent.add_rag_tool(
        collection_name="custom-collection-1",
        tool_name="document_qa",
        description="Use this tool to ask questions about the documents in the collection."
    )
    
    agent.add_sql_tool(
        db_name="custom-db-1",
        tool_name="database_query",
        description="Use this tool to query the database for information."
    )
    
    # Build the agent
    agent.build()
    
    # Example query
    query = "What information can you find about this topic in both the documents and the database?"
    result = agent.run(query)
    print("\nQuery:", query)
    print("\nResult:", result)
