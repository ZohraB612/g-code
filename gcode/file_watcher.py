#!/usr/bin/env python3
"""
File Watcher for gcode - Automatically detects file changes and triggers analysis.
"""

import os
import time
import threading
import hashlib
from pathlib import Path
from typing import Dict, Set, Callable, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileChangeHandler(FileSystemEventHandler):
    """Handles file system events and triggers appropriate actions."""
    
    def __init__(self, callback: Callable, ignored_patterns: Set[str] = None):
        self.callback = callback
        self.ignored_patterns = ignored_patterns or {
            '__pycache__', '.git', '.venv', 'venv', 'node_modules',
            '*.pyc', '*.pyo', '*.log', '*.tmp', '.DS_Store'
        }
        self.last_modified: Dict[str, float] = {}
        self.debounce_time = 1.0  # Wait 1 second before processing changes
        
    def should_ignore(self, file_path: str) -> bool:
        """Check if file should be ignored based on patterns."""
        path = Path(file_path)
        
        # Check if any part of the path matches ignored patterns
        for part in path.parts:
            if part in self.ignored_patterns:
                return True
                
        # Check file extensions
        for pattern in self.ignored_patterns:
            if pattern.startswith('*') and path.suffix == pattern[1:]:
                return True
                
        return False
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
            
        if self.should_ignore(event.src_path):
            return
            
        # Debounce rapid changes
        current_time = time.time()
        if event.src_path in self.last_modified:
            if current_time - self.last_modified[event.src_path] < self.debounce_time:
                return
                
        self.last_modified[event.src_path] = current_time
        
        logger.info(f"File modified: {event.src_path}")
        self.callback('modified', event.src_path)
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
            
        if self.should_ignore(event.src_path):
            return
            
        logger.info(f"File created: {event.src_path}")
        self.callback('created', event.src_path)
    
    def on_deleted(self, event):
        """Handle file deletion events."""
        if event.is_directory:
            return
            
        if self.should_ignore(event.src_path):
            return
            
        logger.info(f"File deleted: {event.src_path}")
        self.callback('deleted', event.src_path)

class FileWatcher:
    """Main file watcher class that monitors project files."""
    
    def __init__(self, project_path: str, callback: Callable = None):
        self.project_path = Path(project_path).resolve()
        self.callback = callback or self._default_callback
        self.observer = Observer()
        self.handler = FileChangeHandler(self.callback)
        self.is_watching = False
        self.file_hashes: Dict[str, str] = {}
        self.initial_scan_complete = False
        
    def _default_callback(self, event_type: str, file_path: str):
        """Default callback for file changes."""
        print(f"📁 {event_type.title()}: {os.path.relpath(file_path, self.project_path)}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _scan_initial_files(self):
        """Perform initial scan of all files to establish baseline."""
        logger.info("Performing initial file scan...")
        
        for root, dirs, files in os.walk(self.project_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in self.handler.ignored_patterns]
            
            for file in files:
                file_path = os.path.join(root, file)
                if not self.handler.should_ignore(file_path):
                    self.file_hashes[file_path] = self._calculate_file_hash(file_path)
        
        self.initial_scan_complete = True
        logger.info(f"Initial scan complete. Monitoring {len(self.file_hashes)} files.")
    
    def start(self, recursive: bool = True):
        """Start watching for file changes."""
        if self.is_watching:
            logger.warning("File watcher is already running.")
            return
            
        try:
            # Perform initial scan
            self._scan_initial_files()
            
            # Start the observer
            self.observer.schedule(
                self.handler, 
                str(self.project_path), 
                recursive=recursive
            )
            self.observer.start()
            self.is_watching = True
            
            logger.info(f"Started watching: {self.project_path}")
            logger.info("Press Ctrl+C to stop watching.")
            
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
            raise
    
    def stop(self):
        """Stop watching for file changes."""
        if not self.is_watching:
            return
            
        self.observer.stop()
        self.observer.join()
        self.is_watching = False
        logger.info("File watcher stopped.")
    
    def get_changed_files(self) -> Set[str]:
        """Get list of files that have changed since last check."""
        changed_files = set()
        
        for file_path in self.file_hashes:
            if os.path.exists(file_path):
                current_hash = self._calculate_file_hash(file_path)
                if current_hash != self.file_hashes[file_path]:
                    changed_files.add(file_path)
                    self.file_hashes[file_path] = current_hash
            else:
                # File was deleted
                changed_files.add(file_path)
                del self.file_hashes[file_path]
        
        return changed_files
    
    def add_file(self, file_path: str):
        """Add a new file to monitoring."""
        if not self.handler.should_ignore(file_path):
            self.file_hashes[file_path] = self._calculate_file_hash(file_path)
            logger.info(f"Added to monitoring: {file_path}")
    
    def remove_file(self, file_path: str):
        """Remove a file from monitoring."""
        if file_path in self.file_hashes:
            del self.file_hashes[file_path]
            logger.info(f"Removed from monitoring: {file_path}")
    
    def get_status(self) -> Dict:
        """Get current status of the file watcher."""
        return {
            'is_watching': self.is_watching,
            'project_path': str(self.project_path),
            'files_monitored': len(self.file_hashes),
            'initial_scan_complete': self.initial_scan_complete
        }

def create_file_watcher(project_path: str, callback: Callable = None) -> FileWatcher:
    """Factory function to create a file watcher."""
    return FileWatcher(project_path, callback)

if __name__ == "__main__":
    # Example usage
    def on_file_change(event_type: str, file_path: str):
        print(f"🔍 {event_type.upper()}: {file_path}")
        # Here you could trigger gcode analysis, linting, etc.
    
    watcher = create_file_watcher(".", on_file_change)
    
    try:
        watcher.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
        print("\n👋 File watcher stopped.")
