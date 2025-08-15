#!/usr/bin/env python3
"""
AI-Enhanced Indexer for gcode - Advanced semantic understanding.
Features: embeddings, code similarity, intelligent suggestions, and AI-powered analysis.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import hashlib
import pickle
from datetime import datetime
import logging
import re # Added missing import for re

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import AI libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

from .advanced_indexer import AdvancedIndexer, Symbol, create_advanced_indexer

logger = logging.getLogger(__name__)

@dataclass
class CodeEmbedding:
    """Represents a code snippet with its semantic embedding."""
    code: str
    embedding: np.ndarray
    symbol_name: str
    file_path: str
    line_number: int
    context: str
    similarity_score: float = 0.0

@dataclass
class CodeSuggestion:
    """AI-generated code improvement suggestion."""
    original_code: str
    suggested_code: str
    explanation: str
    confidence: float
    improvement_type: str  # 'performance', 'security', 'readability', 'best_practice'
    priority: str  # 'low', 'medium', 'high', 'critical'

@dataclass
class CodeAnalysis:
    """Comprehensive AI-powered code analysis."""
    file_path: str
    overall_score: float
    suggestions: List[CodeSuggestion]
    complexity_analysis: Dict[str, Any]
    security_analysis: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    maintainability_score: float
    technical_debt: float

class AIEnhancedIndexer(AdvancedIndexer):
    """
            AI-enhanced indexer that provides advanced capabilities:
    - Semantic code understanding with embeddings
    - AI-powered code suggestions and improvements
    - Code similarity detection
    - Intelligent refactoring recommendations
    - Security and performance analysis
    """
    
    def __init__(self, project_root: str = ".", db_path: str = ".gcode_ai_index.db"):
        super().__init__(project_root, db_path)
        
        # AI capabilities
        self.embeddings_model = None
        self.openai_client = None
        self.gemini_model = None
        
        # Initialize AI components
        self._init_ai_components()
        
        # Enhanced storage
        self.embeddings_cache = {}
        self.similarity_matrix = {}
        self.analysis_cache = {}
        
        # AI configuration
        self.ai_config = {
            'use_embeddings': True,
            'use_openai': OPENAI_AVAILABLE,
            'use_gemini': GEMINI_AVAILABLE,
            'embedding_dimension': 768,
            'similarity_threshold': 0.7,
            'max_suggestions': 10
        }
    
    def _init_ai_components(self):
        """Initialize AI components for enhanced indexing."""
        # Ensure environment variables are loaded
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        # Initialize embeddings model
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Sentence transformers embeddings model loaded")
            except Exception as e:
                logger.warning(f"⚠️  Could not load embeddings model: {e}")
        
        # Initialize OpenAI client
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            try:
                openai.api_key = os.getenv('OPENAI_API_KEY')
                self.openai_client = openai
                logger.info("✅ OpenAI client initialized")
            except Exception as e:
                logger.warning(f"⚠️  Could not initialize OpenAI: {e}")
        
        # Initialize Gemini model
        if GEMINI_AVAILABLE and os.getenv('GEMINI_API_KEY'):
            try:
                genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                logger.info("✅ Gemini model initialized")
            except Exception as e:
                logger.warning(f"⚠️  Could not initialize Gemini: {e}")
    
    def generate_code_embedding(self, code: str, context: str = "") -> np.ndarray:
        """Generate semantic embedding for code snippet."""
        if not self.embeddings_model:
            return np.zeros(self.ai_config['embedding_dimension'])
        
        try:
            # Combine code and context for better semantic understanding
            text = f"{code}\n{context}".strip()
            embedding = self.embeddings_model.encode(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.ai_config['embedding_dimension'])
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_similar_code(self, query_code: str, context: str = "", threshold: float = None) -> List[CodeEmbedding]:
        """Find semantically similar code in the codebase."""
        if not self.embeddings_model:
            return []
        
        threshold = threshold or self.ai_config['similarity_threshold']
        query_embedding = self.generate_code_embedding(query_code, context)
        
        similar_code = []
        
        # Search through cached embeddings
        for symbol_id, embedding_data in self.embeddings_cache.items():
            similarity = self.calculate_similarity(query_embedding, embedding_data['embedding'])
            
            if similarity >= threshold:
                similar_code.append(CodeEmbedding(
                    code=embedding_data['code'],
                    embedding=embedding_data['embedding'],
                    symbol_name=embedding_data['symbol_name'],
                    file_path=embedding_data['file_path'],
                    line_number=embedding_data['line_number'],
                    context=embedding_data['context'],
                    similarity_score=similarity
                ))
        
        # Sort by similarity score
        similar_code.sort(key=lambda x: x.similarity_score, reverse=True)
        return similar_code
    
    def generate_code_suggestions(self, code: str, context: str = "", language: str = "python") -> List[CodeSuggestion]:
        """Generate AI-powered code improvement suggestions."""
        suggestions = []
        
        # Use OpenAI for code suggestions
        if self.openai_client:
            try:
                prompt = f"""
                Analyze this {language} code and provide improvement suggestions:
                
                Code:
                {code}
                
                Context: {context}
                
                Provide suggestions for:
                1. Performance improvements
                2. Security enhancements
                3. Readability improvements
                4. Best practices
                5. Code smells and technical debt
                
                Format each suggestion as:
                - Type: [performance|security|readability|best_practice]
                - Priority: [low|medium|high|critical]
                - Explanation: [detailed explanation]
                - Suggested Code: [improved version]
                - Confidence: [0.0-1.0]
                """
                
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.3
                )
                
                # Parse response and create suggestions
                content = response.choices[0].message.content
                suggestions.extend(self._parse_openai_suggestions(content, code))
                
            except Exception as e:
                logger.error(f"Error generating OpenAI suggestions: {e}")
        
        # Use Gemini as fallback
        elif self.gemini_model:
            try:
                prompt = f"""
                Analyze this {language} code and provide improvement suggestions:
                
                Code:
                {code}
                
                Context: {context}
                
                Focus on performance, security, readability, and best practices.
                """
                
                response = self.gemini_model.generate_content(prompt)
                suggestions.extend(self._parse_gemini_suggestions(response.text, code))
                
            except Exception as e:
                logger.error(f"Error generating Gemini suggestions: {e}")
        
        return suggestions[:self.ai_config['max_suggestions']]
    
    def _parse_openai_suggestions(self, content: str, original_code: str) -> List[CodeSuggestion]:
        """Parse OpenAI response into structured suggestions."""
        suggestions = []
        
        # Simple parsing - in production, use more sophisticated parsing
        lines = content.split('\n')
        current_suggestion = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('- Type:'):
                if current_suggestion:
                    suggestions.append(CodeSuggestion(**current_suggestion))
                current_suggestion = {'original_code': original_code}
                current_suggestion['improvement_type'] = line.split(':', 1)[1].strip()
            elif line.startswith('- Priority:'):
                current_suggestion['priority'] = line.split(':', 1)[1].strip()
            elif line.startswith('- Explanation:'):
                current_suggestion['explanation'] = line.split(':', 1)[1].strip()
            elif line.startswith('- Suggested Code:'):
                current_suggestion['suggested_code'] = line.split(':', 1)[1].strip()
            elif line.startswith('- Confidence:'):
                try:
                    current_suggestion['confidence'] = float(line.split(':', 1)[1].strip())
                except:
                    current_suggestion['confidence'] = 0.7
        
        if current_suggestion:
            suggestions.append(CodeSuggestion(**current_suggestion))
        
        return suggestions
    
    def _parse_gemini_suggestions(self, content: str, original_code: str) -> List[CodeSuggestion]:
        """Parse Gemini response into structured suggestions."""
        # Similar parsing logic for Gemini
        return self._parse_openai_suggestions(content, original_code)
    
    def analyze_code_quality(self, file_path: str) -> CodeAnalysis:
        """Perform comprehensive AI-powered code quality analysis."""
        if file_path in self.analysis_cache:
            return self.analysis_cache[file_path]
        
        # Get file symbols
        symbols = self.get_file_symbols(file_path)
        if not symbols:
            return CodeAnalysis(
                file_path=file_path,
                overall_score=0.0,
                suggestions=[],
                complexity_analysis={},
                security_analysis={},
                performance_analysis={},
                maintainability_score=0.0,
                technical_debt=0.0
            )
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            content = ""
        
        # Generate suggestions
        suggestions = self.generate_code_suggestions(content, f"File: {file_path}")
        
        # Calculate metrics
        complexity_score = self._calculate_complexity_score(symbols)
        security_score = self._calculate_security_score(content, symbols)
        performance_score = self._calculate_performance_score(content, symbols)
        maintainability_score = self._calculate_maintainability_score(symbols)
        technical_debt = self._calculate_technical_debt(symbols, suggestions)
        
        # Overall score (weighted average)
        overall_score = (
            complexity_score * 0.25 +
            security_score * 0.25 +
            performance_score * 0.2 +
            maintainability_score * 0.3
        )
        
        analysis = CodeAnalysis(
            file_path=file_path,
            overall_score=overall_score,
            suggestions=suggestions,
            complexity_analysis={'score': complexity_score, 'details': self._analyze_complexity(symbols)},
            security_analysis={'score': security_score, 'details': self._analyze_security(content)},
            performance_analysis={'score': performance_score, 'details': self._analyze_performance(content)},
            maintainability_score=maintainability_score,
            technical_debt=technical_debt
        )
        
        # Cache the analysis
        self.analysis_cache[file_path] = analysis
        return analysis
    
    def _calculate_complexity_score(self, symbols: List[Symbol]) -> float:
        """Calculate complexity score based on cyclomatic complexity."""
        if not symbols:
            return 1.0
        
        total_complexity = sum(s.complexity for s in symbols)
        avg_complexity = total_complexity / len(symbols)
        
        # Score based on complexity thresholds
        if avg_complexity <= 3:
            return 1.0
        elif avg_complexity <= 5:
            return 0.8
        elif avg_complexity <= 8:
            return 0.6
        elif avg_complexity <= 12:
            return 0.4
        else:
            return 0.2
    
    def _calculate_security_score(self, content: str, symbols: List[Symbol]) -> float:
        """Calculate security score based on common vulnerabilities."""
        score = 1.0
        
        # Check for common security issues
        security_patterns = {
            'sql_injection': [r'execute\(', r'exec\(', r'eval\('],
            'hardcoded_secrets': [r'password\s*=', r'secret\s*=', r'api_key\s*='],
            'unsafe_deserialization': [r'pickle\.loads', r'yaml\.load'],
            'path_traversal': [r'\.\./', r'\.\.\\'],
            'command_injection': [r'subprocess\.call', r'os\.system']
        }
        
        for issue, patterns in security_patterns.items():
            for pattern in patterns:
                if len(re.findall(pattern, content, re.IGNORECASE)) > 0:
                    score -= 0.2
        
        return max(0.0, score)
    
    def _calculate_performance_score(self, content: str, symbols: List[Symbol]) -> float:
        """Calculate performance score based on code patterns."""
        score = 1.0
        
        # Check for performance anti-patterns
        performance_patterns = {
            'nested_loops': [r'for.*for', r'while.*while'],
            'inefficient_data_structures': [r'\.append\(.*\)', r'\.extend\(.*\)'],
            'memory_leaks': [r'global\s+', r'nonlocal\s+'],
            'expensive_operations': [r'\.sort\(\)', r'\.reverse\(\)']
        }
        
        for issue, patterns in performance_patterns.items():
            for pattern in patterns:
                if len(re.findall(pattern, content, re.IGNORECASE)) > 0:
                    score -= 0.15
        
        return max(0.0, score)
    
    def _calculate_maintainability_score(self, symbols: List[Symbol]) -> float:
        """Calculate maintainability score based on code structure."""
        if not symbols:
            return 1.0
        
        # Factors affecting maintainability
        total_symbols = len(symbols)
        functions = [s for s in symbols if s.kind == 'function']
        classes = [s for s in symbols if s.kind == 'class']
        
        # Function length (approximate)
        avg_function_length = total_symbols / max(len(functions), 1)
        
        # Class complexity
        avg_class_complexity = sum(s.complexity for s in classes) / max(len(classes), 1)
        
        # Calculate score
        score = 1.0
        
        if avg_function_length > 50:
            score -= 0.3
        if avg_class_complexity > 10:
            score -= 0.3
        if total_symbols > 100:
            score -= 0.2
        
        return max(0.0, score)
    
    def _calculate_technical_debt(self, symbols: List[Symbol], suggestions: List[CodeSuggestion]) -> float:
        """Calculate technical debt based on code quality issues."""
        debt = 0.0
        
        # Complexity debt
        high_complexity = [s for s in symbols if s.complexity > 10]
        debt += len(high_complexity) * 0.5
        
        # Suggestion debt
        critical_suggestions = [s for s in suggestions if s.priority == 'critical']
        high_suggestions = [s for s in suggestions if s.priority == 'high']
        debt += len(critical_suggestions) * 1.0 + len(high_suggestions) * 0.5
        
        return debt
    
    def _analyze_complexity(self, symbols: List[Symbol]) -> Dict[str, Any]:
        """Detailed complexity analysis."""
        if not symbols:
            return {}
        
        complexity_ranges = {
            'low': [s for s in symbols if s.complexity <= 3],
            'medium': [s for s in symbols if 3 < s.complexity <= 8],
            'high': [s for s in symbols if 8 < s.complexity <= 15],
            'critical': [s for s in symbols if s.complexity > 15]
        }
        
        return {
            'total_symbols': len(symbols),
            'average_complexity': sum(s.complexity for s in symbols) / len(symbols),
            'complexity_distribution': {k: len(v) for k, v in complexity_ranges.items()},
            'most_complex_symbols': sorted(symbols, key=lambda x: x.complexity, reverse=True)[:5]
        }
    
    def _analyze_security(self, content: str) -> Dict[str, Any]:
        """Detailed security analysis."""
        # This would be enhanced with more sophisticated security scanning
        return {
            'vulnerabilities_found': 0,
            'security_score': self._calculate_security_score(content, []),
            'recommendations': []
        }
    
    def _analyze_performance(self, content: str) -> Dict[str, Any]:
        """Detailed performance analysis."""
        # This would be enhanced with more sophisticated performance analysis
        return {
            'performance_issues': 0,
            'performance_score': self._calculate_performance_score(content, []),
            'recommendations': []
        }
    
    def enhanced_search(self, query: str, context: str = "", use_semantics: bool = True) -> List[Dict[str, Any]]:
        """Enhanced search combining traditional and semantic search."""
        results = []
        
        # Traditional search
        traditional_results = self.search_symbols(query)
        for symbol in traditional_results:
            results.append({
                'symbol': symbol,
                'score': 1.0,
                'search_type': 'traditional',
                'explanation': f"Exact match for '{query}'"
            })
        
        # Semantic search if enabled
        if use_semantics and self.embeddings_model:
            semantic_results = self.find_similar_code(query, context)
            for code_embedding in semantic_results:
                results.append({
                    'symbol': code_embedding,
                    'score': code_embedding.similarity_score,
                    'search_type': 'semantic',
                    'explanation': f"Semantically similar to '{query}' (similarity: {code_embedding.similarity_score:.2f})"
                })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def get_code_insights(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive insights about a file."""
        analysis = self.analyze_code_quality(file_path)
        symbols = self.get_file_symbols(file_path)
        
        # Generate embeddings for symbols
        symbol_embeddings = []
        for symbol in symbols[:10]:  # Limit to first 10 for performance
            if hasattr(symbol, 'signature'):
                embedding = self.generate_code_embedding(symbol.signature, symbol.docstring)
                symbol_embeddings.append({
                    'name': symbol.name,
                    'kind': symbol.kind,
                    'embedding': embedding.tolist(),
                    'complexity': symbol.complexity
                })
        
        return {
            'file_path': file_path,
            'analysis': analysis,
            'symbols_count': len(symbols),
            'symbol_embeddings': symbol_embeddings,
            'recommendations': [s.explanation for s in analysis.suggestions],
            'quality_metrics': {
                'overall_score': analysis.overall_score,
                'complexity_score': analysis.complexity_analysis.get('score', 0),
                'security_score': analysis.security_analysis.get('score', 0),
                'performance_score': analysis.performance_analysis.get('score', 0),
                'maintainability_score': analysis.maintainability_score,
                'technical_debt': analysis.technical_debt
            }
        }
    
    def export_enhanced_index(self, output_file: str = ".gcode_ai_index_export.json") -> str:
        """Export enhanced index with AI analysis."""
        export_data = {
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'index_version': '2.0.0',
                'ai_enhanced': True,
                'embeddings_available': EMBEDDINGS_AVAILABLE,
                'openai_available': OPENAI_AVAILABLE,
                'gemini_available': GEMINI_AVAILABLE
            },
            'ai_analysis': {},
            'embeddings': {},
            'similarity_matrix': {}
        }
        
        # Export AI analysis for each file
        for file_path in self.analysis_cache:
            export_data['ai_analysis'][file_path] = self.get_code_insights(file_path)
        
        # Export embeddings
        for symbol_id, embedding_data in self.embeddings_cache.items():
            export_data['embeddings'][symbol_id] = {
                'code': embedding_data['code'],
                'embedding': embedding_data['embedding'].tolist(),
                'symbol_name': embedding_data['symbol_name'],
                'file_path': embedding_data['file_path']
            }
        
        output_path = self.project_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"💾 Enhanced AI index exported to: {output_path}")
        return str(output_path)
    
    def get_ai_capabilities_status(self) -> Dict[str, Any]:
        """Get current status of AI capabilities."""
        # Ensure environment variables are loaded
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        status = {
            'sentence_transformers': {
                'available': EMBEDDINGS_AVAILABLE and self.embeddings_model is not None,
                'model': 'all-MiniLM-L6-v2' if self.embeddings_model else None,
                'dimension': self.ai_config['embedding_dimension'] if self.embeddings_model else None
            },
            'openai': {
                'available': OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY') and self.openai_client is not None,
                'api_key_configured': bool(os.getenv('OPENAI_API_KEY')),
                'client_initialized': self.openai_client is not None
            },
            'gemini': {
                'available': GEMINI_AVAILABLE and os.getenv('GEMINI_API_KEY') and self.gemini_model is not None,
                'api_key_configured': bool(os.getenv('GEMINI_API_KEY')),
                'model_initialized': self.gemini_model is not None
            }
        }
        
        return status

def create_ai_enhanced_indexer(project_root: str = ".") -> AIEnhancedIndexer:
    """Factory function to create an AI-enhanced indexer."""
    return AIEnhancedIndexer(project_root)

if __name__ == "__main__":
    # Example usage
    indexer = create_ai_enhanced_indexer(".")
    
    # Index the codebase
    stats = indexer.index_codebase()
    print(f"AI-enhanced indexing complete: {stats}")
    
    # Test AI features
    if indexer.embeddings_model:
        print("✅ Embeddings model available")
        similar_code = indexer.find_similar_code("def function():", "Python function definition")
        print(f"Found {len(similar_code)} similar code snippets")
    
    if indexer.openai_client or indexer.gemini_model:
        print("✅ AI model available")
        suggestions = indexer.generate_code_suggestions("x = 1 + 1", "Simple addition")
        print(f"Generated {len(suggestions)} code suggestions") 