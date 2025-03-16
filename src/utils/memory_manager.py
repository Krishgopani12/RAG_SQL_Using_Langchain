import os
import json
import csv
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import shutil
from .app_utils import load_config, create_directory
from pyprojroot import here


class MemoryManager:
    """Manages chat history and memory storage."""
    
    _instance = None
    _is_initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not MemoryManager._is_initialized:
            # Load configuration
            config = load_config()
            self.memory_config = config['memory_config']
            
            # Initialize storage
            self._initialize_storage()
            
            # Set up backup if enabled
            if self.memory_config['enable_backup']:
                self._setup_backup()
            
            MemoryManager._is_initialized = True
    
    def _initialize_storage(self) -> None:
        """Initialize storage directories and cleanup old files."""
        # Create main storage directory
        storage_path = here(self.memory_config['storage_path'])
        create_directory(storage_path)
        
        # Clean up old files
        self._cleanup_old_files()
    
    def _setup_backup(self) -> None:
        """Set up backup directory."""
        backup_path = here(self.memory_config['backup_path'])
        create_directory(backup_path)
    
    def _cleanup_old_files(self) -> None:
        """Remove chat history files older than max_history_days."""
        storage_path = here(self.memory_config['storage_path'])
        max_days = self.memory_config['max_history_days']
        cutoff_date = datetime.now() - timedelta(days=max_days)
        
        for filename in os.listdir(storage_path):
            if not filename.endswith(self.memory_config['file_format']):
                continue
            
            file_path = os.path.join(storage_path, filename)
            file_date = datetime.strptime(filename.split('.')[0], '%Y-%m-%d')
            
            if file_date < cutoff_date:
                os.remove(file_path)
    
    def _get_current_file_path(self) -> str:
        """Get the path for today's chat history file."""
        filename = f"{datetime.now().strftime('%Y-%m-%d')}.{self.memory_config['file_format']}"
        return os.path.join(here(self.memory_config['storage_path']), filename)
    
    def save_chat_history(self, history: List[Tuple[str, str]], session_id: Optional[str] = None) -> None:
        """
        Save chat history to storage.
        
        Args:
            history (List[Tuple[str, str]]): List of (user_message, bot_response) tuples
            session_id (Optional[str]): Unique session identifier
        """
        file_path = self._get_current_file_path()
        timestamp = datetime.now().isoformat()
        
        if self.memory_config['file_format'] == 'csv':
            self._save_to_csv(file_path, history, timestamp, session_id)
        else:  # json
            self._save_to_json(file_path, history, timestamp, session_id)
        
        # Perform backup if needed
        self._backup_if_needed()
    
    def _save_to_csv(self, file_path: str, history: List[Tuple[str, str]], 
                     timestamp: str, session_id: Optional[str]) -> None:
        """Save chat history to CSV file."""
        mode = 'a' if os.path.exists(file_path) else 'w'
        with open(file_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if mode == 'w':
                writer.writerow(['timestamp', 'session_id', 'user_message', 'bot_response'])
            
            for user_msg, bot_resp in history:
                writer.writerow([timestamp, session_id or '', user_msg, bot_resp])
    
    def _save_to_json(self, file_path: str, history: List[Tuple[str, str]], 
                      timestamp: str, session_id: Optional[str]) -> None:
        """Save chat history to JSON file."""
        data = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        for user_msg, bot_resp in history:
            entry = {
                'timestamp': timestamp,
                'session_id': session_id or '',
                'user_message': user_msg,
                'bot_response': bot_resp
            }
            data.append(entry)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _backup_if_needed(self) -> None:
        """Create backup based on configured frequency."""
        if not self.memory_config['enable_backup']:
            return
        
        now = datetime.now()
        backup_path = here(self.memory_config['backup_path'])
        storage_path = here(self.memory_config['storage_path'])
        
        if self.memory_config['backup_frequency'] == 'daily':
            backup_dir = os.path.join(backup_path, now.strftime('%Y-%m-%d'))
        elif self.memory_config['backup_frequency'] == 'weekly':
            backup_dir = os.path.join(backup_path, f"week_{now.strftime('%Y-%W')}")
        else:  # monthly
            backup_dir = os.path.join(backup_path, f"month_{now.strftime('%Y-%m')}")
        
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            # Copy all files from storage to backup
            for filename in os.listdir(storage_path):
                if filename.endswith(self.memory_config['file_format']):
                    src = os.path.join(storage_path, filename)
                    dst = os.path.join(backup_dir, filename)
                    shutil.copy2(src, dst)
    
    def get_chat_history(self, days: int = 1, session_id: Optional[str] = None) -> List[Dict]:
        """
        Retrieve chat history for the specified period.
        
        Args:
            days (int): Number of days of history to retrieve
            session_id (Optional[str]): Filter by session ID
            
        Returns:
            List[Dict]: List of chat history entries
        """
        history = []
        start_date = datetime.now() - timedelta(days=days)
        storage_path = here(self.memory_config['storage_path'])
        
        for filename in os.listdir(storage_path):
            if not filename.endswith(self.memory_config['file_format']):
                continue
                
            file_date = datetime.strptime(filename.split('.')[0], '%Y-%m-%d')
            if file_date >= start_date:
                file_path = os.path.join(storage_path, filename)
                if self.memory_config['file_format'] == 'csv':
                    history.extend(self._read_from_csv(file_path, session_id))
                else:  # json
                    history.extend(self._read_from_json(file_path, session_id))
        
        return sorted(history, key=lambda x: x['timestamp'])
    
    def _read_from_csv(self, file_path: str, session_id: Optional[str]) -> List[Dict]:
        """Read chat history from CSV file."""
        history = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if session_id is None or row['session_id'] == session_id:
                    history.append({
                        'timestamp': row['timestamp'],
                        'session_id': row['session_id'],
                        'user_message': row['user_message'],
                        'bot_response': row['bot_response']
                    })
        return history
    
    def _read_from_json(self, file_path: str, session_id: Optional[str]) -> List[Dict]:
        """Read chat history from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if session_id is None:
                return data
            return [entry for entry in data if entry['session_id'] == session_id] 