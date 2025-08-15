#!/usr/bin/env python3
"""
Cursor-like Codebase Indexer for gcode
Provides sophisticated indexing similar to Cursor's codebase understanding capabilities.
"""

import os
import ast
import re
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from datetime import datetime
import threading
import time
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Symbol:
    """Represents a code symbol (function, class, variable, etc.)"""
    name: str
    kind: str  # 'function', 'class', 'variable', 'import', 'decorator'
    file_path: str
    line_number: int
    column: int
    end_line: int
    end_column: int
    signature: str = ""
    docstring: str = ""
    parent: str = ""  # For methods, nested functions, etc.
    visibility: str = "public"  # public, private, protected
    complexity: int = 0
    dependencies: List[str] = None
    references: List[Tuple[str, int, int]] = None  # (file_path, line, col)
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.references is None:
            self.references = []

@dataclass
class FileIndex:
    """Represents the index of a single file"""
    path: str
    absolute_path: str
    file_type: str
    size: int
    lines: int
    last_modified: float
    hash: str
    symbols: List[Symbol]
    imports: List[str]
    exports: List[str]
    dependencies: List[str]
    dependents: List[str]
    complexity: int
    language_features: Dict[str, Any]

class CursorIndexer:
    """
    Advanced codebase indexer that provides Cursor-like capabilities:
    - Fast symbol resolution
    - Semantic search
    - Dependency tracking
    - Real-time updates
    - Cross-file references
    """
    
    def __init__(self, project_root: str = ".", db_path: str = ".gcode_index.db"):
        self.project_root = Path(project_root).resolve()
        self.db_path = self.project_root / db_path
        self.index_lock = threading.RLock()
        
        # File type handlers
        self.language_handlers = {
            'python': PythonHandler(),
            'javascript': JavaScriptHandler(),
            'typescript': TypeScriptHandler(),
            'java': JavaHandler(),
            'cpp': CppHandler(),
            'rust': RustHandler(),
            'go': GoHandler(),
        }
        
        # Initialize database
        self._init_database()
        
        # File watching
        self.watched_files = set()
        self.file_hashes = {}
        self.last_index_update = 0
        
        # Index statistics
        self.stats = {
            'total_files': 0,
            'total_symbols': 0,
            'last_index_time': 0,
            'index_version': '2.0.0'
        }
    
    def _init_database(self):
        """Initialize SQLite database for fast symbol lookup."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line_number INTEGER NOT NULL,
                    column INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    end_column INTEGER NOT NULL,
                    signature TEXT,
                    docstring TEXT,
                    parent TEXT,
                    visibility TEXT,
                    complexity INTEGER,
                    dependencies TEXT,
                    symbol_references TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    absolute_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    lines INTEGER NOT NULL,
                    last_modified REAL NOT NULL,
                    hash TEXT NOT NULL,
                    complexity INTEGER,
                    language_features TEXT,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dependencies (
                    file_path TEXT NOT NULL,
                    dependency_path TEXT NOT NULL,
                    dependency_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (file_path, dependency_path)
                )
            """)
            
            # Create indexes for fast lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_type ON files(file_type)")
            
            conn.commit()
    
    def index_codebase(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Index the entire codebase for fast symbol resolution and search.
        
        Args:
            force_reindex: If True, reindex all files even if unchanged
            
        Returns:
            Indexing statistics and summary
        """
        with self.index_lock:
            logger.info("🔍 Starting comprehensive codebase indexing...")
            start_time = time.time()
            
            # Get all files to index
            files_to_index = self._get_files_to_index(force_reindex)
            
            if not files_to_index and not force_reindex:
                logger.info("📚 All files are up to date, using existing index")
                return self._get_index_stats()
            
            # Clear old index data
            self._clear_old_index()
            
            # Index each file
            total_symbols = 0
            for file_path in files_to_index:
                try:
                    symbols = self._index_file(file_path)
                    total_symbols += len(symbols)
                    self._store_file_index(file_path, symbols)
                except Exception as e:
                    logger.error(f"Error indexing {file_path}: {e}")
            
            # Build dependency graph
            self._build_dependency_graph()
            
            # Update statistics
            self.stats.update({
                'total_files': len(files_to_index),
                'total_symbols': total_symbols,
                'last_index_time': time.time(),
            })
            
            # Save statistics
            self._save_stats()
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Indexing complete: {len(files_to_index)} files, {total_symbols} symbols in {elapsed:.2f}s")
            
            return self._get_index_stats()
    
    def _get_files_to_index(self, force_reindex: bool) -> List[Path]:
        """Get list of files that need indexing."""
        files_to_index = []
        
        for file_path in self.project_root.rglob('*'):
            if not self._should_index_file(file_path):
                continue
            
            if force_reindex or self._file_needs_indexing(file_path):
                files_to_index.append(file_path)
        
        return files_to_index
    
    def _should_index_file(self, file_path: Path) -> bool:
        """Check if a file should be indexed."""
        if not file_path.is_file():
            return False
        
        # Check ignore patterns
        ignore_patterns = {
            '__pycache__', '.git', '.vscode', '.idea', 'node_modules',
            'build', 'dist', 'target', 'venv', '.env', '.pyc', '.pyo',
            '.so', '.dll', '.exe', '.log', '.tmp', '.cache'
        }
        
        for pattern in ignore_patterns:
            if pattern in str(file_path):
                return False
        
        # Check file extensions
        supported_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.cc',
            '.cxx', '.h', '.hpp', '.rs', '.go', '.rb', '.php', '.cs'
        }
        
        return file_path.suffix.lower() in supported_extensions
    
    def _file_needs_indexing(self, file_path: Path) -> bool:
        """Check if a file needs re-indexing."""
        try:
            current_hash = self._calculate_file_hash(file_path)
            stored_hash = self.file_hashes.get(str(file_path))
            
            if stored_hash != current_hash:
                self.file_hashes[str(file_path)] = current_hash
                return True
            
            return False
        except:
            return True
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.sha256(content).hexdigest()
        except:
            return str(file_path.stat().st_mtime)
    
    def _index_file(self, file_path: Path) -> List[Symbol]:
        """Index a single file and extract all symbols."""
        file_type = self._get_file_type(file_path)
        handler = self.language_handlers.get(file_type)
        
        if not handler:
            logger.warning(f"No handler for file type: {file_type} ({file_path})")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            symbols = handler.parse_file(str(file_path), content)
            logger.debug(f"Indexed {file_path}: {len(symbols)} symbols")
            return symbols
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return []
    
    def _get_file_type(self, file_path: Path) -> str:
        """Determine the file type based on extension."""
        ext = file_path.suffix.lower()
        
        type_mapping = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.h': 'cpp',
            '.hpp': 'cpp',
            '.rs': 'rust',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp'
        }
        
        return type_mapping.get(ext, 'unknown')
    
    def _store_file_index(self, file_path: Path, symbols: List[Symbol]):
        """Store file index and symbols in the database."""
        with sqlite3.connect(self.db_path) as conn:
            # Store file info
            file_info = {
                'path': str(file_path.relative_to(self.project_root)),
                'absolute_path': str(file_path),
                'file_type': self._get_file_type(file_path),
                'size': file_path.stat().st_size,
                'lines': len(symbols) if symbols else 0,
                'last_modified': file_path.stat().st_mtime,
                'hash': self.file_hashes.get(str(file_path), ''),
                'complexity': sum(s.complexity for s in symbols),
                'language_features': json.dumps({})
            }
            
            conn.execute("""
                INSERT OR REPLACE INTO files 
                (path, absolute_path, file_type, size, lines, last_modified, hash, complexity, language_features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(file_info.values()))
            
            # Store symbols
            for symbol in symbols:
                symbol_data = asdict(symbol)
                symbol_data['dependencies'] = json.dumps(symbol.dependencies)
                symbol_data['symbol_references'] = json.dumps(symbol.references)
                
                conn.execute("""
                    INSERT OR REPLACE INTO symbols 
                    (name, kind, file_path, line_number, column, end_line, end_column, 
                     signature, docstring, parent, visibility, complexity, dependencies, symbol_references)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol.name, symbol.kind, symbol.file_path, symbol.line_number, symbol.column,
                    symbol.end_line, symbol.end_column, symbol.signature, symbol.docstring,
                    symbol.parent, symbol.visibility, symbol.complexity, 
                    json.dumps(symbol.dependencies), json.dumps(symbol.references)
                ))
            
            conn.commit()
    
    def _clear_old_index(self):
        """Clear old index data from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM symbols")
            conn.execute("DELETE FROM files")
            conn.execute("DELETE FROM dependencies")
            conn.commit()
    
    def _build_dependency_graph(self):
        """Build dependency graph between files."""
        logger.info("🔗 Building dependency graph...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Get all import statements
            imports = conn.execute("""
                SELECT file_path, dependencies FROM symbols 
                WHERE kind = 'import' AND dependencies != '[]'
            """).fetchall()
            
            for file_path, deps_json in imports:
                try:
                    deps = json.loads(deps_json)
                    for dep in deps:
                        conn.execute("""
                            INSERT OR REPLACE INTO dependencies 
                            (file_path, dependency_path, dependency_type)
                            VALUES (?, ?, ?)
                        """, (file_path, dep, 'import'))
                except:
                    continue
            
            conn.commit()
    
    def _save_stats(self):
        """Save index statistics."""
        stats_file = self.project_root / ".gcode_index_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
    
    def _get_index_stats(self) -> Dict[str, Any]:
        """Get current index statistics."""
        return self.stats.copy()
    
    def search_symbols(self, query: str, kind: str = None, file_path: str = None, limit: int = None) -> List[Symbol]:
        """
        Search for symbols by name, kind, or file.
        
        Args:
            query: Search query (symbol name or partial match)
            kind: Filter by symbol kind
            file_path: Filter by file path
            limit: Maximum number of results to return
            
        Returns:
            List of matching symbols
        """
        with sqlite3.connect(self.db_path) as conn:
            sql = """
                SELECT * FROM symbols 
                WHERE name LIKE ? 
            """
            params = [f"%{query}%"]
            
            if kind:
                sql += " AND kind = ?"
                params.append(kind)
            
            if file_path:
                sql += " AND file_path LIKE ?"
                params.append(f"%{file_path}%")
            
            sql += " ORDER BY name, file_path"
            
            if limit:
                sql += f" LIMIT {limit}"
            
            results = conn.execute(sql, params).fetchall()
            
            symbols = []
            for row in results:
                symbol = Symbol(
                    name=row[1], kind=row[2], file_path=row[3],
                    line_number=row[4], column=row[5], end_line=row[6],
                    end_column=row[7], signature=row[8], docstring=row[9],
                    parent=row[10], visibility=row[11], complexity=row[12]
                )
                
                # Parse JSON fields
                try:
                    if row[13]:  # dependencies
                        symbol.dependencies = json.loads(row[13])
                    if row[14]:  # symbol_references
                        symbol.references = json.loads(row[14])
                except:
                    pass
                
                symbols.append(symbol)
            
            return symbols
    
    def get_symbol_definition(self, symbol_name: str, file_path: str = None) -> Optional[Symbol]:
        """
        Get the definition of a specific symbol.
        
        Args:
            symbol_name: Name of the symbol to find
            file_path: Optional file path to search in
            
        Returns:
            Symbol definition or None if not found
        """
        symbols = self.search_symbols(symbol_name, file_path=file_path)
        return symbols[0] if symbols else None
    
    def get_file_symbols(self, file_path: str) -> List[Symbol]:
        """
        Get all symbols in a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of symbols in the file
        """
        return self.search_symbols("", file_path=file_path)
    
    def get_dependencies(self, file_path: str) -> List[str]:
        """
        Get dependencies of a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of dependency paths
        """
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute("""
                SELECT dependency_path FROM dependencies 
                WHERE file_path = ?
            """, (file_path,)).fetchall()
            
            return [row[0] for row in results]
    
    def get_dependents(self, file_path: str) -> List[str]:
        """
        Get files that depend on a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of dependent file paths
        """
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute("""
                SELECT file_path FROM dependencies 
                WHERE dependency_path = ?
            """, (file_path,)).fetchall()
            
            return [row[0] for row in results]
    
    def semantic_search(self, query: str, context: str = None) -> List[Symbol]:
        """
        Perform semantic search for symbols based on context and meaning.
        
        Args:
            query: Natural language query
            context: Optional context (e.g., current file, function)
            
        Returns:
            List of relevant symbols
        """
        # This is a simplified semantic search - could be enhanced with embeddings
        query_lower = query.lower()
        
        # Extract key terms
        terms = re.findall(r'\b\w+\b', query_lower)
        
        # Search for symbols that match the terms
        all_symbols = self.search_symbols("")
        relevant_symbols = []
        
        for symbol in all_symbols:
            score = 0
            
            # Name matching
            if any(term in symbol.name.lower() for term in terms):
                score += 3
            
            # Kind matching
            if any(term in symbol.kind.lower() for term in terms):
                score += 2
            
            # Docstring matching
            if symbol.docstring and any(term in symbol.docstring.lower() for term in terms):
                score += 1
            
            # Parent context matching
            if context and symbol.parent and context.lower() in symbol.parent.lower():
                score += 2
            
            if score > 0:
                symbol.score = score
                relevant_symbols.append(symbol)
        
        # Sort by relevance score
        relevant_symbols.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
        return relevant_symbols
    
    def get_project_overview(self) -> Dict[str, Any]:
        """
        Get a comprehensive overview of the indexed project.
        
        Returns:
            Project overview with statistics and structure
        """
        with sqlite3.connect(self.db_path) as conn:
            # File statistics
            file_stats = conn.execute("""
                SELECT file_type, COUNT(*) as count, SUM(complexity) as total_complexity
                FROM files GROUP BY file_type
            """).fetchall()
            
            # Symbol statistics
            symbol_stats = conn.execute("""
                SELECT kind, COUNT(*) as count, AVG(complexity) as avg_complexity
                FROM symbols GROUP BY kind
            """).fetchall()
            
            # Top complex files
            complex_files = conn.execute("""
                SELECT path, complexity FROM files 
                ORDER BY complexity DESC LIMIT 10
            """).fetchall()
            
            # Top complex symbols
            complex_symbols = conn.execute("""
                SELECT name, kind, file_path, complexity FROM symbols 
                ORDER BY complexity DESC LIMIT 10
            """).fetchall()
        
        return {
            'index_stats': self.stats,
            'file_statistics': {row[0]: {'count': row[1], 'complexity': row[2]} for row in file_stats},
            'symbol_statistics': {row[0]: {'count': row[1], 'avg_complexity': row[2]} for row in symbol_stats},
            'most_complex_files': [{'path': row[0], 'complexity': row[1]} for row in complex_files],
            'most_complex_symbols': [{'name': row[0], 'kind': row[1], 'file': row[2], 'complexity': row[3]} for row in complex_symbols]
        }
    
    def export_index(self, output_file: str = ".gcode_index_export.json") -> str:
        """
        Export the complete index to a JSON file.
        
        Args:
            output_file: Output file path
            
        Returns:
            Path to the exported file
        """
        export_data = {
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'index_version': self.stats['index_version'],
                'project_root': str(self.project_root)
            },
            'overview': self.get_project_overview(),
            'files': {},
            'symbols': []
        }
        
        with sqlite3.connect(self.db_path) as conn:
            # Export files
            files = conn.execute("SELECT * FROM files").fetchall()
            for file_row in files:
                file_path = file_row[0]
                export_data['files'][file_path] = {
                    'absolute_path': file_row[1],
                    'file_type': file_row[2],
                    'size': file_row[3],
                    'lines': file_row[4],
                    'last_modified': file_row[5],
                    'hash': file_row[6],
                    'complexity': file_row[7],
                    'language_features': json.loads(file_row[8]) if file_row[8] else {}
                }
            
            # Export symbols
            symbols = conn.execute("SELECT * FROM symbols").fetchall()
            for symbol_row in symbols:
                symbol_data = {
                    'name': symbol_row[1],
                    'kind': symbol_row[2],
                    'file_path': symbol_row[3],
                    'line_number': symbol_row[4],
                    'column': symbol_row[5],
                    'end_line': symbol_row[6],
                    'end_column': symbol_row[7],
                    'signature': symbol_row[8],
                    'docstring': symbol_row[9],
                    'parent': symbol_row[10],
                    'visibility': symbol_row[11],
                    'complexity': symbol_row[12],
                    'dependencies': json.loads(symbol_row[13]) if symbol_row[13] else [],
                    'references': json.loads(symbol_row[14]) if symbol_row[14] else []
                }
                export_data['symbols'].append(symbol_data)
        
        output_path = self.project_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"💾 Index exported to: {output_path}")
        return str(output_path)

# Language-specific handlers
class PythonHandler:
    """Handles Python file parsing and symbol extraction."""
    
    def parse_file(self, file_path: str, content: str) -> List[Symbol]:
        symbols = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    symbol = self._extract_function(node, file_path, content)
                    symbols.append(symbol)
                elif isinstance(node, ast.ClassDef):
                    symbol = self._extract_class(node, file_path, content)
                    symbols.append(symbol)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        symbol = self._extract_import(alias, node, file_path, content)
                        symbols.append(symbol)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        symbol = self._extract_import_from(alias, node, file_path, content)
                        symbols.append(symbol)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            symbol = self._extract_variable(target, node, file_path, content)
                            symbols.append(symbol)
        
        except SyntaxError:
            logger.warning(f"Syntax error in {file_path}")
        
        return symbols
    
    def _extract_function(self, node: ast.FunctionDef, file_path: str, content: str) -> Symbol:
        lines = content.split('\n')
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
        
        # Get function signature
        args = [arg.arg for arg in node.args.args]
        signature = f"def {node.name}({', '.join(args)})"
        
        # Calculate complexity
        complexity = self._calculate_complexity(node)
        
        # Determine visibility
        visibility = "private" if node.name.startswith('_') else "public"
        
        return Symbol(
            name=node.name,
            kind='function',
            file_path=file_path,
            line_number=start_line,
            column=node.col_offset,
            end_line=end_line,
            end_column=0,  # Would need more sophisticated parsing
            signature=signature,
            docstring=ast.get_docstring(node) or "",
            parent="",
            visibility=visibility,
            complexity=complexity
        )
    
    def _extract_class(self, node: ast.ClassDef, file_path: str, content: str) -> Symbol:
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
        
        # Get class bases
        bases = [self._get_base_name(base) for base in node.bases]
        signature = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
        
        # Determine visibility
        visibility = "private" if node.name.startswith('_') else "public"
        
        return Symbol(
            name=node.name,
            kind='class',
            file_path=file_path,
            line_number=start_line,
            column=node.col_offset,
            end_line=end_line,
            end_column=0,
            signature=signature,
            docstring=ast.get_docstring(node) or "",
            parent="",
            visibility=visibility,
            complexity=0
        )
    
    def _extract_import(self, alias: ast.alias, node: ast.Import, file_path: str, content: str) -> Symbol:
        return Symbol(
            name=alias.asname or alias.name,
            kind='import',
            file_path=file_path,
            line_number=node.lineno,
            column=node.col_offset,
            end_line=node.lineno,
            end_column=0,
            signature=f"import {alias.name}",
            docstring="",
            parent="",
            visibility="public",
            complexity=0,
            dependencies=[alias.name]
        )
    
    def _extract_import_from(self, alias: ast.alias, node: ast.ImportFrom, file_path: str, content: str) -> Symbol:
        module = node.module or ""
        return Symbol(
            name=alias.asname or alias.name,
            kind='import',
            file_path=file_path,
            line_number=node.lineno,
            column=node.col_offset,
            end_line=node.lineno,
            end_column=0,
            signature=f"from {module} import {alias.name}",
            docstring="",
            parent="",
            visibility="public",
            complexity=0,
            dependencies=[module]
        )
    
    def _extract_variable(self, target: ast.Name, node: ast.Assign, file_path: str, content: str) -> Symbol:
        return Symbol(
            name=target.id,
            kind='variable',
            file_path=file_path,
            line_number=node.lineno,
            column=node.col_offset,
            end_line=node.lineno,
            end_column=0,
            signature=f"{target.id} = ...",
            docstring="",
            parent="",
            visibility="private" if target.id.startswith('_') else "public",
            complexity=0
        )
    
    def _get_base_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        return "object"
    
    def _calculate_complexity(self, node) -> int:
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity

class JavaScriptHandler:
    """Handles JavaScript/TypeScript file parsing."""
    
    def parse_file(self, file_path: str, content: str) -> List[Symbol]:
        symbols = []
        
        # Extract functions
        function_pattern = r'(?:function\s+(\w+)|(\w+)\s*[:=]\s*(?:async\s+)?function|(\w+)\s*[:=]\s*\([^)]*\)\s*=>)'
        for match in re.finditer(function_pattern, content):
            name = next(n for n in match.groups() if n)
            symbols.append(Symbol(
                name=name,
                kind='function',
                file_path=file_path,
                line_number=content[:match.start()].count('\n') + 1,
                column=match.start() - content.rfind('\n', 0, match.start()) - 1,
                end_line=0,
                end_column=0,
                signature=f"function {name}",
                docstring="",
                parent="",
                visibility="public",
                complexity=0
            ))
        
        # Extract classes
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            symbols.append(Symbol(
                name=match.group(1),
                kind='class',
                file_path=file_path,
                line_number=content[:match.start()].count('\n') + 1,
                column=match.start() - content.rfind('\n', 0, match.start()) - 1,
                end_line=0,
                end_column=0,
                signature=f"class {match.group(1)}",
                docstring="",
                parent="",
                visibility="public",
                complexity=0
            ))
        
        return symbols

class TypeScriptHandler(JavaScriptHandler):
    """Handles TypeScript file parsing - extends JavaScript handler with TS-specific features."""
    
    def parse_file(self, file_path: str, content: str) -> List[Symbol]:
        # Get base JavaScript symbols
        symbols = super().parse_file(file_path, content)
        
        # Add TypeScript-specific symbols
        # Extract interfaces
        interface_pattern = r'interface\s+(\w+)'
        for match in re.finditer(interface_pattern, content):
            symbols.append(Symbol(
                name=match.group(1),
                kind='interface',
                file_path=file_path,
                line_number=content[:match.start()].count('\n') + 1,
                column=match.start() - content.rfind('\n', 0, match.start()) - 1,
                end_line=0,
                end_column=0,
                signature=f"interface {match.group(1)}",
                docstring="",
                parent="",
                visibility="public",
                complexity=0
            ))
        
        # Extract type aliases
        type_pattern = r'type\s+(\w+)\s*='
        for match in re.finditer(type_pattern, content):
            symbols.append(Symbol(
                name=match.group(1),
                kind='type',
                file_path=file_path,
                line_number=content[:match.start()].count('\n') + 1,
                column=match.start() - content.rfind('\n', 0, match.start()) - 1,
                end_line=0,
                end_column=0,
                signature=f"type {match.group(1)}",
                docstring="",
                parent="",
                visibility="public",
                complexity=0
            ))
        
        return symbols

class JavaHandler:
    """Handles Java file parsing."""
    
    def parse_file(self, file_path: str, content: str) -> List[Symbol]:
        symbols = []
        
        # Extract classes
        class_pattern = r'(?:public\s+)?class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            symbols.append(Symbol(
                name=match.group(1),
                kind='class',
                file_path=file_path,
                line_number=content[:match.start()].count('\n') + 1,
                column=0,
                end_line=0,
                end_column=0,
                signature=f"class {match.group(1)}",
                docstring="",
                parent="",
                visibility="public",
                complexity=0
            ))
        
        return symbols

class CppHandler:
    """Handles C++ file parsing."""
    
    def parse_file(self, file_path: str, content: str) -> List[Symbol]:
        symbols = []
        
        # Extract functions
        function_pattern = r'(\w+)\s+\w+\s*\([^)]*\)\s*{'
        for match in re.finditer(function_pattern, content):
            symbols.append(Symbol(
                name=match.group(1),
                kind='function',
                file_path=file_path,
                line_number=content[:match.start()].count('\n') + 1,
                column=0,
                end_line=0,
                end_column=0,
                signature=f"{match.group(1)}(...)",
                docstring="",
                parent="",
                visibility="public",
                complexity=0
            ))
        
        return symbols

class RustHandler:
    """Handles Rust file parsing."""
    
    def parse_file(self, file_path: str, content: str) -> List[Symbol]:
        symbols = []
        
        # Extract functions
        function_pattern = r'fn\s+(\w+)\s*\('
        for match in re.finditer(function_pattern, content):
            symbols.append(Symbol(
                name=match.group(1),
                kind='function',
                file_path=file_path,
                line_number=content[:match.start()].count('\n') + 1,
                column=0,
                end_line=0,
                end_column=0,
                signature=f"fn {match.group(1)}(...)",
                docstring="",
                parent="",
                visibility="public",
                complexity=0
            ))
        
        return symbols

class GoHandler:
    """Handles Go file parsing."""
    
    def parse_file(self, file_path: str, content: str) -> List[Symbol]:
        symbols = []
        
        # Extract functions
        function_pattern = r'func\s+(\w+)\s*\('
        for match in re.finditer(function_pattern, content):
            symbols.append(Symbol(
                name=match.group(1),
                kind='function',
                file_path=file_path,
                line_number=content[:match.start()].count('\n') + 1,
                column=0,
                end_line=0,
                end_column=0,
                signature=f"func {match.group(1)}(...)",
                docstring="",
                parent="",
                visibility="public",
                complexity=0
            ))
        
        return symbols

def create_cursor_indexer(project_root: str = ".") -> CursorIndexer:
    """Factory function to create a Cursor-like indexer."""
    return CursorIndexer(project_root)

if __name__ == "__main__":
    # Example usage
    indexer = create_cursor_indexer(".")
    
    # Index the codebase
    stats = indexer.index_codebase()
    print(f"Indexing complete: {stats}")
    
    # Search for symbols
    symbols = indexer.search_symbols("function")
    print(f"Found {len(symbols)} symbols matching 'function'")
    
    # Get project overview
    overview = indexer.get_project_overview()
    print(f"Project overview: {overview}") 