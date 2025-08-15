#!/usr/bin/env python3
"""
Deep Codebase Analyzer for gcode - Builds semantic understanding of entire projects.
This provides the "agentic search" capabilities that make gcode so powerful.
"""

import os
import ast
import re
from pathlib import Path
import json
from typing import Dict, List, Set, Optional, Any
from datetime import datetime

class CodebaseAnalyzer:
    """Analyzes entire codebases to build a comprehensive knowledge graph."""
    
    def __init__(self, project_root="."):
        self.project_root = Path(project_root).resolve()
        self.knowledge_graph = {}
        self.file_hashes = {}  # Track file changes
        self.analysis_timestamp = None
        
        # File patterns to analyze
        self.analyze_patterns = {
            'python': '*.py',
            'javascript': '*.js',
            'typescript': '*.ts',
            'java': '*.java',
            'cpp': '*.cpp',
            'c': '*.c',
            'rust': '*.rs',
            'go': '*.go',
            'config': ['*.json', '*.yaml', '*.yml', '*.toml', '*.ini'],
            'docs': ['*.md', '*.rst', '*.txt']
        }
        
        # Patterns to ignore
        self.ignore_patterns = {
            'venv', '__pycache__', '.git', '.vscode', '.idea',
            'node_modules', 'build', 'dist', 'target', '*.pyc',
            '.pytest_cache', '.coverage', '*.log'
        }
    
    def should_ignore(self, file_path: Path) -> bool:
        """Check if file should be ignored during analysis."""
        file_str = str(file_path)
        
        # Check ignore patterns
        for pattern in self.ignore_patterns:
            if pattern in file_str:
                return True
        
        # Check if it's a binary file
        if file_path.suffix in ['.pyc', '.pyo', '.so', '.dll', '.exe']:
            return True
            
        return False
    
    def analyze(self, force_reanalysis: bool = False) -> Dict:
        """
        Analyzes the entire codebase and builds a comprehensive knowledge graph.
        
        Args:
            force_reanalysis: If True, reanalyze all files even if unchanged
            
        Returns:
            The complete knowledge graph
        """
        print("🔍 Building deep codebase understanding...")
        
        # Check if we need to reanalyze
        if not force_reanalysis and self._can_use_cached_analysis():
            print("📚 Using cached knowledge graph...")
            return self.knowledge_graph
        
        # Clear previous analysis
        self.knowledge_graph = {}
        self.file_hashes = {}
        
        # Analyze all supported file types
        total_files = 0
        for file_type, patterns in self.analyze_patterns.items():
            if isinstance(patterns, str):
                patterns = [patterns]
            
            for pattern in patterns:
                files = list(self.project_root.rglob(pattern))
                for file_path in files:
                    if not self.should_ignore(file_path):
                        self.analyze_file(file_path, file_type)
                        total_files += 1
        
        # Build relationships and dependencies
        self._map_dependencies()
        self._build_architecture_overview()
        self._analyze_project_patterns()
        
        # Save analysis
        self.analysis_timestamp = datetime.now().isoformat()
        self.save()
        
        print(f"✅ Deep analysis complete: {total_files} files mapped")
        print(f"📊 Knowledge graph: {len(self.knowledge_graph)} files analyzed")
        
        return self.knowledge_graph
    
    def analyze_file(self, file_path: Path, file_type: str):
        """Analyzes a single file and adds it to the knowledge graph."""
        try:
            # Calculate file hash for change detection
            file_hash = self._calculate_file_hash(file_path)
            self.file_hashes[str(file_path)] = file_hash
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Initialize file info
            file_info = {
                'path': str(file_path.relative_to(self.project_root)),
                'absolute_path': str(file_path),
                'file_type': file_type,
                'size': len(content),
                'lines': content.count('\n') + 1,
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                'analysis_timestamp': self.analysis_timestamp
            }
            
            # Type-specific analysis
            if file_type == 'python':
                file_info.update(self._analyze_python_file(content))
            elif file_type in ['javascript', 'typescript']:
                file_info.update(self._analyze_js_file(content))
            elif file_type in ['java', 'cpp', 'c']:
                file_info.update(self._analyze_compiled_file(content, file_type))
            elif file_type == 'config':
                file_info.update(self._analyze_config_file(content, file_path))
            elif file_type == 'docs':
                file_info.update(self._analyze_doc_file(content, file_path))
            
            # Add to knowledge graph
            self.knowledge_graph[str(file_path)] = file_info
            
        except Exception as e:
            print(f"⚠️  Error analyzing {file_path}: {e}")
    
    def _analyze_python_file(self, content: str) -> Dict:
        """Deep analysis of Python files using AST."""
        try:
            tree = ast.parse(content)
            
            analysis = {
                'imports': [],
                'from_imports': [],
                'functions': [],
                'classes': [],
                'variables': [],
                'docstrings': [],
                'complexity': 0,
                'dependencies': []
            }
            
            # Analyze AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append({
                            'module': alias.name,
                            'alias': alias.asname or alias.name
                        })
                elif isinstance(node, ast.ImportFrom):
                    analysis['from_imports'].append({
                        'module': node.module or '',
                        'names': [n.name for n in node.names],
                        'level': node.level
                    })
                elif isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'defaults': len(node.args.defaults),
                        'docstring': ast.get_docstring(node),
                        'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
                        'complexity': self._calculate_complexity(node)
                    }
                    analysis['functions'].append(func_info)
                    analysis['complexity'] += func_info['complexity']
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'bases': [self._get_base_name(base) for base in node.bases],
                        'methods': [],
                        'class_variables': [],
                        'docstring': ast.get_docstring(node),
                        'decorators': [self._get_decorator_name(d) for d in node.decorator_list]
                    }
                    
                    # Analyze class body
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info['methods'].append(item.name)
                        elif isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    class_info['class_variables'].append(target.id)
                    
                    analysis['classes'].append(class_info)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            analysis['variables'].append(target.id)
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
                    analysis['docstrings'].append(node.value.s)
            
            return analysis
            
        except SyntaxError:
            # Handle syntax errors gracefully
            return {
                'imports': [],
                'from_imports': [],
                'functions': [],
                'classes': [],
                'variables': [],
                'docstrings': [],
                'complexity': 0,
                'dependencies': [],
                'syntax_error': True
            }
    
    def _analyze_js_file(self, content: str) -> Dict:
        """Basic analysis of JavaScript/TypeScript files."""
        analysis = {
            'imports': [],
            'exports': [],
            'functions': [],
            'classes': [],
            'variables': []
        }
        
        # Simple regex-based analysis for JS/TS
        import_pattern = r'import\s+(?:{[^}]*}|\*\s+as\s+\w+|\w+)\s+from\s+[\'"]([^\'"]+)[\'"]'
        export_pattern = r'export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)'
        function_pattern = r'(?:function\s+(\w+)|(\w+)\s*[:=]\s*(?:async\s+)?function|(\w+)\s*[:=]\s*\([^)]*\)\s*=>)'
        class_pattern = r'class\s+(\w+)'
        
        # Extract patterns
        analysis['imports'] = re.findall(import_pattern, content)
        analysis['exports'] = re.findall(export_pattern, content)
        analysis['functions'] = [f for f in re.findall(function_pattern, content) if f]
        analysis['classes'] = re.findall(class_pattern, content)
        
        return analysis
    
    def _analyze_compiled_file(self, content: str, file_type: str) -> Dict:
        """Basic analysis of compiled language files."""
        analysis = {
            'file_type': file_type,
            'imports': [],
            'functions': [],
            'classes': []
        }
        
        # Language-specific patterns
        if file_type == 'java':
            class_pattern = r'class\s+(\w+)'
            method_pattern = r'(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?\w+\s+(\w+)\s*\('
            analysis['classes'] = re.findall(class_pattern, content)
            analysis['functions'] = re.findall(method_pattern, content)
        elif file_type in ['cpp', 'c']:
            function_pattern = r'(\w+)\s+\w+\s*\([^)]*\)\s*{'
            class_pattern = r'class\s+(\w+)'
            analysis['functions'] = re.findall(function_pattern, content)
            analysis['classes'] = re.findall(class_pattern, content)
        
        return analysis
    
    def _analyze_config_file(self, content: str, file_path: Path) -> Dict:
        """Analyze configuration files."""
        analysis = {
            'config_type': file_path.suffix[1:],
            'keys': [],
            'sections': []
        }
        
        try:
            if file_path.suffix == '.json':
                config = json.loads(content)
                analysis['keys'] = list(config.keys())
            elif file_path.suffix in ['.yaml', '.yml']:
                # Simple YAML key extraction
                lines = content.split('\n')
                for line in lines:
                    if ':' in line and not line.strip().startswith('#'):
                        key = line.split(':')[0].strip()
                        if key:
                            analysis['keys'].append(key)
        except:
            pass
        
        return analysis
    
    def _analyze_doc_file(self, content: str, file_path: Path) -> Dict:
        """Analyze documentation files."""
        analysis = {
            'doc_type': file_path.suffix[1:],
            'headings': [],
            'links': [],
            'code_blocks': 0
        }
        
        # Extract headings
        heading_pattern = r'^#{1,6}\s+(.+)$'
        analysis['headings'] = re.findall(heading_pattern, content, re.MULTILINE)
        
        # Extract links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        analysis['links'] = re.findall(link_pattern, content)
        
        # Count code blocks
        code_block_pattern = r'```[\s\S]*?```'
        analysis['code_blocks'] = len(re.findall(code_block_pattern, content))
        
        return analysis
    
    def _map_dependencies(self):
        """Maps dependencies between files in the knowledge graph."""
        print("🔗 Mapping file dependencies...")
        
        for file_path, file_info in self.knowledge_graph.items():
            file_info['dependencies'] = []
            file_info['dependents'] = []
            
            # Map Python imports to files
            if file_info.get('file_type') == 'python':
                for imp in file_info.get('imports', []):
                    module_name = imp.get('module', '')
                    if module_name:
                        # Find files that might match this import
                        for other_path, other_info in self.knowledge_graph.items():
                            if other_path != file_path:
                                if self._is_import_match(module_name, other_path):
                                    file_info['dependencies'].append(other_path)
                                    if 'dependents' not in other_info:
                                        other_info['dependents'] = []
                                    other_info['dependents'].append(file_path)
    
    def _is_import_match(self, module_name: str, file_path: str) -> bool:
        """Check if an import matches a file path."""
        file_name = Path(file_path).stem
        
        # Direct module name match
        if module_name == file_name:
            return True
        
        # Check if file is in a package that matches the import
        if '.' in module_name:
            parts = module_name.split('.')
            file_parts = Path(file_path).parts
            
            # Check if the file path contains the module parts
            for i, part in enumerate(parts):
                if i < len(file_parts) and file_parts[i] == part:
                    continue
                else:
                    return False
            return True
        
        return False
    
    def _build_architecture_overview(self):
        """Builds a high-level architecture overview of the project."""
        print("🏗️  Building architecture overview...")
        
        architecture = {
            'entry_points': [],
            'main_modules': [],
            'packages': [],
            'test_files': [],
            'config_files': [],
            'documentation': []
        }
        
        for file_path, file_info in self.knowledge_graph.items():
            path = file_info['path']
            
            # Identify entry points
            if path in ['main.py', '__main__.py', 'app.py', 'run.py']:
                architecture['entry_points'].append(path)
            
            # Identify main modules
            elif path.endswith('.py') and not any(x in path for x in ['test_', '_test', 'tests/']):
                architecture['main_modules'].append(path)
            
            # Identify packages
            elif path.endswith('__init__.py'):
                architecture['packages'].append(path)
            
            # Identify test files
            elif 'test' in path.lower() or 'tests/' in path:
                architecture['test_files'].append(path)
            
            # Identify config files
            elif file_info.get('file_type') == 'config':
                architecture['config_files'].append(path)
            
            # Identify documentation
            elif file_info.get('file_type') == 'docs':
                architecture['documentation'].append(path)
        
        # Add architecture to knowledge graph
        self.knowledge_graph['__architecture__'] = {
            'type': 'architecture_overview',
            'timestamp': self.analysis_timestamp,
            'overview': architecture,
            'total_files': len(self.knowledge_graph),
            'python_files': len([f for f in self.knowledge_graph.values() if f.get('file_type') == 'python']),
            'test_coverage': len(architecture['test_files']) / max(len(architecture['main_modules']), 1)
        }
    
    def _analyze_project_patterns(self):
        """Analyzes project patterns and provides insights."""
        print("🔍 Analyzing project patterns...")
        
        patterns = {
            'frameworks': [],
            'testing_frameworks': [],
            'build_tools': [],
            'code_quality': {},
            'complexity_metrics': {}
        }
        
        # Detect frameworks and tools
        for file_path, file_info in self.knowledge_graph.items():
            content = ''
            try:
                with open(file_info['absolute_path'], 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except:
                continue
            
            # Framework detection
            if 'django' in content.lower():
                patterns['frameworks'].append('Django')
            if 'flask' in content.lower():
                patterns['frameworks'].append('Flask')
            if 'fastapi' in content.lower():
                patterns['frameworks'].append('FastAPI')
            if 'pytest' in content.lower():
                patterns['testing_frameworks'].append('pytest')
            if 'unittest' in content.lower():
                patterns['testing_frameworks'].append('unittest')
            if 'makefile' in content.lower():
                patterns['build_tools'].append('Make')
            if 'dockerfile' in content.lower():
                patterns['build_tools'].append('Docker')
        
        # Remove duplicates
        patterns['frameworks'] = list(set(patterns['frameworks']))
        patterns['testing_frameworks'] = list(set(patterns['testing_frameworks']))
        patterns['build_tools'] = list(set(patterns['build_tools']))
        
        # Add patterns to knowledge graph
        self.knowledge_graph['__patterns__'] = {
            'type': 'project_patterns',
            'timestamp': self.analysis_timestamp,
            'patterns': patterns
        }
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate a simple hash of file content."""
        try:
            stat = file_path.stat()
            return f"{stat.st_mtime}_{stat.st_size}"
        except:
            return "unknown"
    
    def _can_use_cached_analysis(self) -> bool:
        """Check if we can use cached analysis."""
        cache_file = self.project_root / ".gcode_knowledge_graph.json"
        if not cache_file.exists():
            return False
        
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            
            # Check if files have changed
            for file_path, file_info in cached.items():
                if file_path.startswith('__'):  # Skip metadata
                    continue
                
                actual_path = self.project_root / file_path
                if not actual_path.exists():
                    return False  # File was deleted
                
                current_hash = self._calculate_file_hash(actual_path)
                if current_hash != file_info.get('hash', ''):
                    return False  # File changed
            
            # Use cached data
            self.knowledge_graph = cached
            return True
            
        except:
            return False
    
    def _get_decorator_name(self, node) -> str:
        """Extract decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
        return "unknown"
    
    def _get_base_name(self, node) -> str:
        """Extract base class name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        return "unknown"
    
    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def save(self, output_file=".gcode_knowledge_graph.json"):
        """Saves the knowledge graph to a file."""
        output_path = self.project_root / output_file
        
        # Add metadata
        save_data = {
            '__metadata__': {
                'version': '1.0',
                'created': self.analysis_timestamp,
                'total_files': len(self.knowledge_graph),
                'analyzer_version': '1.0'
            },
            **self.knowledge_graph
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"💾 Knowledge graph saved to: {output_path}")
    
    def load(self, input_file=".gcode_knowledge_graph.json"):
        """Loads the knowledge graph from a file."""
        input_path = self.project_root / input_file
        
        if input_path.exists():
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Remove metadata
                if '__metadata__' in data:
                    del data['__metadata__']
                
                self.knowledge_graph = data
                print(f"📚 Knowledge graph loaded from: {input_path}")
                return True
            except Exception as e:
                print(f"❌ Error loading knowledge graph: {e}")
        
        return False
    
    def query(self, question: str) -> str:
        """
        Query the knowledge graph with natural language questions.
        This provides the "agentic search" capability.
        """
        question = question.lower()
        
        # Simple query patterns - could be enhanced with NLP
        if "what functions are in" in question:
            file_name = question.split("what functions are in")[-1].strip()
            for file_path, file_info in self.knowledge_graph.items():
                if file_name in file_path and file_info.get('functions'):
                    funcs = [f['name'] for f in file_info['functions']]
                    return f"Functions in {file_path}: {', '.join(funcs)}"
        
        elif "what are the dependencies of" in question:
            file_name = question.split("what are the dependencies of")[-1].strip()
            for file_path, file_info in self.knowledge_graph.items():
                if file_name in file_path:
                    deps = file_info.get('dependencies', [])
                    return f"Dependencies of {file_path}: {', '.join(deps) if deps else 'None'}"
        
        elif "show me the architecture" in question:
            arch = self.knowledge_graph.get('__architecture__', {})
            if arch:
                overview = arch.get('overview', {})
                return f"Project Architecture:\n" + \
                       f"- Entry points: {', '.join(overview.get('entry_points', []))}\n" + \
                       f"- Main modules: {len(overview.get('main_modules', []))}\n" + \
                       f"- Packages: {len(overview.get('packages', []))}\n" + \
                       f"- Test files: {len(overview.get('test_files', []))}\n" + \
                       f"- Test coverage: {arch.get('test_coverage', 0):.1%}"
        
        elif "what frameworks are used" in question:
            patterns = self.knowledge_graph.get('__patterns__', {})
            if patterns:
                frameworks = patterns.get('patterns', {}).get('frameworks', [])
                return f"Frameworks detected: {', '.join(frameworks) if frameworks else 'None detected'}"
        
        return "I can answer questions about:\n" + \
               "- Functions in specific files\n" + \
               "- File dependencies\n" + \
               "- Project architecture\n" + \
               "- Frameworks used\n" + \
               "Try asking about one of these topics!"

def create_analyzer(project_root=".") -> CodebaseAnalyzer:
    """Factory function to create a codebase analyzer."""
    return CodebaseAnalyzer(project_root)

if __name__ == "__main__":
    # Example usage
    analyzer = create_analyzer(".")
    knowledge_graph = analyzer.analyze()
    
    print(f"\n🎯 Knowledge Graph Summary:")
    print(f"Files analyzed: {len(knowledge_graph)}")
    
    # Show some insights
    arch = knowledge_graph.get('__architecture__', {})
    if arch:
        print(f"Python files: {arch.get('overview', {}).get('main_modules', [])}")
    
    # Test querying
    print(f"\n🔍 Sample queries:")
    print(analyzer.query("show me the architecture"))
    print(analyzer.query("what frameworks are used"))
