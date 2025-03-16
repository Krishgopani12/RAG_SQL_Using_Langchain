from pathlib import Path
from dotenv import load_dotenv
import os

def load_environment():
    """Load and validate environment variables."""
    env_path = Path(__file__).parents[2] / '.env'
    load_dotenv(env_path)
    
    # List of required environment variables
    required_vars = ['OPENAI_API_KEY']
    
    # Check for missing environment variables
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        ) 