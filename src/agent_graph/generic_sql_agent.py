from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
import os
import yaml
from pyprojroot import here
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import re

def resolve_env_vars(value: str) -> str:
    """
    Resolve environment variables in a string.
    
    Args:
        value (str): String containing environment variable placeholders
        
    Returns:
        str: String with environment variables resolved
    """
    if not isinstance(value, str):
        return value
        
    pattern = r'\${([^}]+)}'
    matches = re.findall(pattern, value)
    
    resolved = value
    for env_var in matches:
        env_value = os.getenv(env_var)
        if env_value is None:
            raise ValueError(f"Environment variable {env_var} not found")
        resolved = resolved.replace(f"${{{env_var}}}", env_value)
    
    return resolved

class Table(BaseModel):
    """
    Represents a table in the SQL database.

    Attributes:
        name (str): The name of the table in the SQL database.
    """
    name: str = Field(description="Name of table in SQL database.")


class SQLDatabaseAgent:
    """A generic SQL database agent that can work with any SQL database type."""
    
    def __init__(self, db_config: Dict, llm_config: Dict):
        """
        Initialize the SQL database agent.
        
        Args:
            db_config (Dict): Database configuration containing connection details
            llm_config (Dict): Language model configuration
        """
        self.db_config = self._resolve_config(db_config)
        self.llm_config = llm_config
        self.db_type = self.db_config.get('type', 'sqlite')
        self.db = self._create_db_connection()
        self.llm = self._create_llm()
        self.agent_executor = self._create_agent()
        self._engine = None

    def _resolve_config(self, config: Dict) -> Dict:
        """Resolve environment variables in configuration."""
        resolved = {}
        for key, value in config.items():
            if isinstance(value, dict):
                resolved[key] = self._resolve_config(value)
            elif isinstance(value, str):
                resolved[key] = resolve_env_vars(value)
            else:
                resolved[key] = value
        return resolved

    def _create_db_connection(self) -> SQLDatabase:
        """Create database connection based on database type."""
        try:
            if self.db_type == 'sqlite':
                db_path = here(self.db_config['path'])
                self._engine = create_engine(f"sqlite:///{db_path}")
                return SQLDatabase.from_uri(
                    f"sqlite:///{db_path}",
                    sample_rows_in_table_info=2,
                    include_tables=None,
                    view_support=True
                )
            
            elif self.db_type in ['postgres', 'postgresql']:
                conn_str = (
                    f"postgresql://{self.db_config['username']}:{self.db_config['password']}"
                    f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
                )
                self._engine = create_engine(conn_str)
                return SQLDatabase.from_uri(conn_str)
            
            elif self.db_type in ['mysql', 'mariadb']:
                conn_str = (
                    f"mysql+pymysql://{self.db_config['username']}:{self.db_config['password']}"
                    f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
                )
                self._engine = create_engine(conn_str)
                return SQLDatabase.from_uri(conn_str)
            
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
        except Exception as e:
            raise ValueError(f"Failed to create database connection: {str(e)}")

    def _create_llm(self) -> ChatOpenAI:
        """Create LLM instance with specified configuration."""
        return ChatOpenAI(
            model=self.llm_config.get('model', 'gpt-4'),
            temperature=self.llm_config.get('temperature', 0.0),
            max_tokens=self.llm_config.get('max_tokens', 1000)
        )

    def _create_agent(self):
        """Create SQL agent with database toolkit."""
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        return create_sql_agent(
            llm=self.llm,
            toolkit=toolkit,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=10
        )

    def run_query(self, query: str) -> str:
        """
        Run a natural language query against the database.
        
        Args:
            query (str): Natural language query to execute
            
        Returns:
            str: Response from the agent
        """
        try:
            # Check if query is related to travel and contains airport codes
            if self._is_travel_query(query):
                return self._handle_travel_query(query)
            
            # Use invoke instead of run to address deprecation warning
            try:
                result = self.agent_executor.invoke({"input": query})
                
                # Extract the output from the response
                if isinstance(result, dict) and "output" in result:
                    return result["output"]
                elif isinstance(result, str):
                    return result
                else:
                    return f"Received response in unexpected format: {str(result)}"
            except AttributeError:
                # Fallback to direct run method if invoke is not available
                print("Warning: Using deprecated run method as fallback")
                return self.agent_executor.run(input=query)
                
        except ValueError as e:
            error_msg = str(e)
            if "must be real number" in error_msg:
                return self._handle_numeric_type_error(query, error_msg)
            elif "parsing errors" in error_msg.lower() or "parsing llm output" in error_msg.lower():
                return self._handle_parsing_error(query, error_msg)
            else:
                return f"Error executing query: {error_msg}"
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def get_table_info(self) -> str:
        """Get information about all tables in the database."""
        try:
            tables = self.db.get_usable_table_names()
            info = []
            
            for table in tables:
                # Get column information
                with self._engine.connect() as conn:
                    result = conn.execute(text(f"PRAGMA table_info('{table}')"))
                    columns = [dict(row) for row in result]
                
                # Format column information
                column_info = [f"{col['name']} ({col['type']})" for col in columns]
                table_info = f"Table: {table}\nColumns: {', '.join(column_info)}"
                
                # Get sample row count
                try:
                    with self._engine.connect() as conn:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM '{table}'"))
                        count = result.scalar()
                        table_info += f"\nRow count: {count}"
                except Exception as e:
                    table_info += f"\nError getting row count: {str(e)}"
                
                info.append(table_info)
            
            return "\n\n".join(info)
        except Exception as e:
            return f"Error getting table info: {str(e)}"

    def get_table_names(self) -> List[str]:
        """Get list of all table names in the database."""
        return self.db.get_usable_table_names()

    def _is_travel_query(self, query: str) -> bool:
        """Check if the query is related to travel."""
        return re.search(r'\b[A-Z]{3}\b', query) is not None

    def _handle_travel_query(self, query: str) -> str:
        """Handle travel-related queries with special processing."""
        try:
            # Extract airport codes
            airport_codes = re.findall(r'\b[A-Z]{3}\b', query)
            
            # Add guidance for the agent
            enhanced_query = (
                f"{query}\n\nNote: This query involves airport codes: {', '.join(airport_codes)}. "
                f"When working with airport codes, ensure all data is treated as text/string values, "
                f"not as numeric values. Do not attempt numeric operations on airport codes."
            )
            
            # Use invoke instead of run with fallback
            try:
                result = self.agent_executor.invoke({"input": enhanced_query})
                
                # Extract the output from the response
                if isinstance(result, dict) and "output" in result:
                    return result["output"]
                elif isinstance(result, str):
                    return result
                else:
                    return f"Received response in unexpected format: {str(result)}"
            except AttributeError:
                # Fallback to direct run method if invoke is not available
                print("Warning: Using deprecated run method as fallback")
                return self.agent_executor.run(input=enhanced_query)
                
        except Exception as e:
            return (
                f"Error processing travel query: {str(e)}\n\n"
                f"Tip: When querying about flights, please use the standard 3-letter airport codes "
                f"(e.g., JFK for New York, LHR for London Heathrow). The system will treat these as text values."
            )

    def _handle_numeric_type_error(self, query: str, error_msg: str) -> str:
        """Handle numeric type errors."""
        return f"Error: {error_msg}\n\nTip: When querying about numbers, please use the standard format for numbers (e.g., 12345, 12.345)."

    def _handle_parsing_error(self, query: str, error_msg: str) -> str:
        """Handle parsing errors."""
        return f"Error: {error_msg}\n\nTip: Please check the format of your query and try again."


class SQLAgentCallback(BaseCallbackHandler):
    """Callback handler for SQL agent execution."""
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when agent takes an action."""
        print(f"\nAgent action: {action.tool}\nAction input: {action.tool_input}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when agent finishes execution."""
        print(f"\nAgent finished: {finish.return_values}")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM completes a generation."""
        pass


def create_sql_agent_from_config(config_path: str, db_name: str) -> SQLDatabaseAgent:
    """
    Create a SQL agent from configuration file.
    
    Args:
        config_path (str): Path to configuration file
        db_name (str): Name of the database configuration to use
        
    Returns:
        SQLDatabaseAgent: Configured SQL agent
    """
    with open(here(config_path)) as f:
        config = yaml.safe_load(f)

    # Get database configuration
    db_config = None
    sql_config = config['sql_db_config']
    
    # Search for database configuration in all database types
    for db_type in ['sqlite', 'postgres', 'mysql']:
        if db_type in sql_config:
            for db in sql_config[db_type]['databases']:
                if db['name'] == db_name:
                    db_config = db.copy()
                    db_config['type'] = db_type
                    break
            if db_config:
                break
    
    if not db_config:
        raise ValueError(f"Database configuration not found for: {db_name}")

    return SQLDatabaseAgent(
        db_config=db_config,
        llm_config=config['llm_config']
    )


if __name__ == "__main__":
    load_dotenv()
    
    # Example usage
    agent = create_sql_agent_from_config(
        config_path="configs/custom_config.yml",
        db_name="custom-db-1"
    )
    
    # Get database information
    print("Available tables:", agent.get_table_names())
    print("\nTable information:", agent.get_table_info())
    
    # Example query
    query = "What tables are available in the database and what are their relationships?"
    result = agent.run_query(query)
    print("\nQuery result:", result) 