import os
import json
import time
import datetime
from typing import Dict, Any, List, Optional, Union
import uuid

class ParameterLogger:
    """
    Logger for Coded Exposure Photography Tool that records user actions and parameters.
    creates a log file that can be used for parameter sweeping and analysis.
    """
    
    def __init__(self, log_dir: str = "logs", session_id: Optional[str] = None):

        self.log_dir = log_dir
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create or use session ID
        self.session_id = session_id if session_id else str(uuid.uuid4())
        
        # Initialize session log
        self.session_log = {
            "session_id": self.session_id,
            "start_time": datetime.datetime.now().isoformat(),
            "actions": [],
            "parameters": {}
        }
        
        # Create log file path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"session_{timestamp}_{self.session_id[:8]}.json")
        
        # Log the session start
        self.log_action("session_start", {})
        
        print(f"Session logging initialized: {self.log_file}")
    
    def log_action(self, action_type: str, action_data: Dict[str, Any]) -> None:
        action_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": action_type,
            "data": action_data
        }
        
        self.session_log["actions"].append(action_entry)
        
        # Auto-save after each action for safety
        self._save_log()
    
    def log_parameter(self, parameter_name: str, parameter_value: Any) -> None:
        # Store the new parameter value
        self.session_log["parameters"][parameter_name] = parameter_value
        
        # Also log it as a change action for the timeline
        self.log_action("parameter_change", {
            "parameter": parameter_name,
            "value": parameter_value
        })
    
    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        # Update parameters dictionary
        self.session_log["parameters"].update(parameters)
        
        # Log as a single action
        self.log_action("parameters_update", parameters)
    
    def get_session_log(self) -> Dict[str, Any]:
        return self.session_log
    
    def get_parameters(self) -> Dict[str, Any]:
        return self.session_log["parameters"]
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        return self.session_log["actions"]
    
    def _save_log(self) -> None:
        with open(self.log_file, 'w') as f:
            json.dump(self.session_log, f, indent=2)
    
    def save_log(self) -> str:
        self._save_log()
        return self.log_file
    
    def export_log(self, export_path: Optional[str] = None) -> str:
        if export_path is None:
            # Generate a descriptive filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(self.log_dir, f"exported_{timestamp}_{self.session_id[:8]}.json")
        
        # Add export timestamp
        self.session_log["export_time"] = datetime.datetime.now().isoformat()
        
        # Save to export path
        with open(export_path, 'w') as f:
            json.dump(self.session_log, f, indent=2)
        
        return export_path
    
    def close(self) -> None:
        self.session_log["end_time"] = datetime.datetime.now().isoformat()
        self.log_action("session_end", {})
        self._save_log()
        print(f"Session log saved to: {self.log_file}")

# Utility functions for working with logs

def load_log(log_file: str) -> Dict[str, Any]:
    with open(log_file, 'r') as f:
        return json.load(f)

def merge_logs(log_files: List[str], output_file: Optional[str] = None) -> str:
    merged_log = {
        "merged_from": log_files,
        "merge_time": datetime.datetime.now().isoformat(),
        "sessions": []
    }
    
    for log_file in log_files:
        try:
            log_data = load_log(log_file)
            merged_log["sessions"].append(log_data)
        except Exception as e:
            print(f"Error loading log file {log_file}: {e}")
    
    if output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(os.path.dirname(log_files[0]), f"merged_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(merged_log, f, indent=2)
    
    return output_file 