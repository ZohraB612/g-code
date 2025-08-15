#!/usr/bin/env python3
"""
AI-Enhanced CLI for gcode - Advanced intelligent analysis.
Features: semantic search, code suggestions, quality analysis, and AI-powered insights.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import json
from tabulate import tabulate
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .ai_enhanced_indexer import create_ai_enhanced_indexer, CodeAnalysis

class AICLI:
    """Advanced CLI interface for AI-enhanced codebase analysis."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.indexer = create_ai_enhanced_indexer(str(self.project_root))
    
    def run(self, args: List[str]) -> int:
        """Run the AI-enhanced CLI with the given arguments."""
        parser = self._create_parser()
        parsed_args = parser.parse_args(args)
        
        try:
            if parsed_args.command == 'analyze':
                return self._cmd_analyze(parsed_args)
            elif parsed_args.command == 'semantic':
                return self._cmd_semantic(parsed_args)
            elif parsed_args.command == 'suggestions':
                return self._cmd_suggestions(parsed_args)
            elif parsed_args.command == 'quality':
                return self._cmd_quality(parsed_args)
            elif parsed_args.command == 'insights':
                return self._cmd_insights(parsed_args)
            elif parsed_args.command == 'similar':
                return self._cmd_similar(parsed_args)
            elif parsed_args.command == 'export':
                return self._cmd_export(parsed_args)
            elif parsed_args.command == 'status':
                return self._cmd_status(parsed_args)
            else:
                parser.print_help()
                return 1
                
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user")
            return 1
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser for AI-enhanced features."""
        parser = argparse.ArgumentParser(
            description="AI-Enhanced codebase analyzer for gcode",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Analyze code quality with AI
  gcode-ai analyze gcode/agent.py
  
  # Semantic search for code
  gcode-ai semantic "authentication function"
  
  # Get AI-powered code suggestions
  gcode-ai suggestions "def process_data(data): pass"
  
  # Analyze overall code quality
  gcode-ai quality gcode/agent.py --detailed
  
  # Get comprehensive insights
  gcode-ai insights gcode/agent.py
  
  # Find similar code
  gcode-ai similar "def calculate_total(items):"
  
  # Export AI-enhanced index
  gcode-ai export --output ai_analysis.json
  
  # Check AI capabilities status
  gcode-ai status
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available AI commands')
        
        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='AI-powered code quality analysis')
        analyze_parser.add_argument('file_path', help='Path to the file to analyze')
        analyze_parser.add_argument('--detailed', '-d', action='store_true',
                                   help='Show detailed analysis')
        
        # Semantic search command
        semantic_parser = subparsers.add_parser('semantic', help='Semantic code search')
        semantic_parser.add_argument('query', help='Natural language query')
        semantic_parser.add_argument('--context', '-c', help='Additional context')
        analyze_parser.add_argument('--limit', '-l', type=int, default=10,
                                   help='Maximum number of results')
        
        # Suggestions command
        suggestions_parser = subparsers.add_parser('suggestions', help='AI code improvement suggestions')
        suggestions_parser.add_argument('code', help='Code to analyze')
        suggestions_parser.add_argument('--context', '-c', help='Code context')
        suggestions_parser.add_argument('--language', '-l', default='python',
                                       help='Programming language')
        
        # Quality command
        quality_parser = subparsers.add_parser('quality', help='Code quality metrics')
        quality_parser.add_argument('file_path', help='Path to the file')
        quality_parser.add_argument('--detailed', '-d', action='store_true',
                                   help='Show detailed metrics')
        
        # Insights command
        insights_parser = subparsers.add_parser('insights', help='Comprehensive code insights')
        insights_parser.add_argument('file_path', help='Path to the file')
        
        # Similar code command
        similar_parser = subparsers.add_parser('similar', help='Find similar code')
        similar_parser.add_argument('code', help='Code snippet to find similar code for')
        similar_parser.add_argument('--context', '-c', help='Code context')
        similar_parser.add_argument('--threshold', '-t', type=float, default=0.7,
                                   help='Similarity threshold (0.0-1.0)')
        
        # Export command
        export_parser = subparsers.add_parser('export', help='Export AI-enhanced analysis')
        export_parser.add_argument('--output', '-o', default='.gcode_ai_export.json',
                                  help='Output file path')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Check AI capabilities status')
        
        return parser
    
    def _cmd_analyze(self, args) -> int:
        """Handle the analyze command."""
        file_path = args.file_path
        
        print(f"🤖 AI-Powered Code Analysis for: {file_path}")
        print("=" * 60)
        
        # Perform AI analysis
        analysis = self.indexer.analyze_code_quality(file_path)
        
        # Display results
        print(f"📊 Overall Quality Score: {analysis.overall_score:.2f}/1.0")
        print()
        
        # Quality breakdown
        print("🎯 Quality Metrics:")
        print(f"  • Complexity Score: {analysis.complexity_analysis.get('score', 0):.2f}")
        print(f"  • Security Score: {analysis.security_analysis.get('score', 0):.2f}")
        print(f"  • Performance Score: {analysis.performance_analysis.get('score', 0):.2f}")
        print(f"  • Maintainability Score: {analysis.maintainability_score:.2f}")
        print(f"  • Technical Debt: {analysis.technical_debt:.2f}")
        print()
        
        # Suggestions
        if analysis.suggestions:
            print(f"💡 AI Suggestions ({len(analysis.suggestions)}):")
            for i, suggestion in enumerate(analysis.suggestions, 1):
                print(f"  {i}. [{suggestion.priority.upper()}] {suggestion.improvement_type}")
                print(f"     {suggestion.explanation}")
                if args.detailed:
                    print(f"     Suggested: {suggestion.suggested_code}")
                    print(f"     Confidence: {suggestion.confidence:.2f}")
                print()
        else:
            print("✅ No AI suggestions generated")
        
        if args.detailed:
            # Show detailed complexity analysis
            complexity_details = analysis.complexity_analysis.get('details', {})
            if complexity_details:
                print("🔍 Detailed Complexity Analysis:")
                print(f"  • Total Symbols: {complexity_details.get('total_symbols', 0)}")
                print(f"  • Average Complexity: {complexity_details.get('average_complexity', 0):.2f}")
                
                distribution = complexity_details.get('complexity_distribution', {})
                if distribution:
                    print("  • Complexity Distribution:")
                    for level, count in distribution.items():
                        print(f"    - {level.title()}: {count}")
                
                most_complex = complexity_details.get('most_complex_symbols', [])
                if most_complex:
                    print("  • Most Complex Symbols:")
                    for symbol in most_complex[:5]:
                        print(f"    - {symbol.name}: complexity {symbol.complexity}")
        
        return 0
    
    def _cmd_semantic(self, args) -> int:
        """Handle the semantic search command."""
        query = args.query
        context = args.context or ""
        
        print(f"🧠 Semantic Search for: {query}")
        if context:
            print(f"📝 Context: {context}")
        print("=" * 50)
        
        # Perform semantic search
        results = self.indexer.enhanced_search(query, context, use_semantics=True)
        
        if not results:
            print("❌ No results found")
            return 0
        
        # Limit results
        results = results[:args.limit]
        
        print(f"✅ Found {len(results)} results:")
        print()
        
        # Prepare table data
        table_data = []
        for result in results:
            symbol = result['symbol']
            table_data.append([
                symbol.name if hasattr(symbol, 'name') else getattr(symbol, 'symbol_name', 'Unknown'),
                result['search_type'],
                f"{result['score']:.3f}",
                getattr(symbol, 'file_path', 'Unknown'),
                getattr(symbol, 'line_number', 'Unknown'),
                result['explanation'][:60] + "..." if len(result['explanation']) > 60 else result['explanation']
            ])
        
        headers = ["Name", "Type", "Score", "File", "Line", "Explanation"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        return 0
    
    def _cmd_suggestions(self, args) -> int:
        """Handle the suggestions command."""
        code = args.code
        context = args.context or ""
        language = args.language
        
        print(f"💡 AI Code Suggestions for {language.upper()}")
        print("=" * 50)
        print(f"Code: {code}")
        if context:
            print(f"Context: {context}")
        print()
        
        # Generate AI suggestions
        suggestions = self.indexer.generate_code_suggestions(code, context, language)
        
        if not suggestions:
            print("❌ No AI suggestions generated")
            print("💡 Make sure you have OpenAI or Gemini API keys configured")
            return 0
        
        print(f"✅ Generated {len(suggestions)} suggestions:")
        print()
        
        for i, suggestion in enumerate(suggestions, 1):
            print(f"🔧 Suggestion {i}: [{suggestion.priority.upper()}] {suggestion.improvement_type}")
            print(f"   📝 {suggestion.explanation}")
            print(f"   💻 Suggested Code: {suggestion.suggested_code}")
            print(f"   🎯 Confidence: {suggestion.confidence:.2f}")
            print()
        
        return 0
    
    def _cmd_quality(self, args) -> int:
        """Handle the quality command."""
        file_path = args.file_path
        
        print(f"📊 Code Quality Analysis for: {file_path}")
        print("=" * 50)
        
        # Get quality metrics
        insights = self.indexer.get_code_insights(file_path)
        metrics = insights['quality_metrics']
        
        # Display metrics
        print("🎯 Quality Scores (0.0 - 1.0):")
        print(f"  • Overall Quality: {metrics['overall_score']:.3f}")
        print(f"  • Complexity: {metrics['complexity_score']:.3f}")
        print(f"  • Security: {metrics['security_score']:.3f}")
        print(f"  • Performance: {metrics['performance_score']:.3f}")
        print(f"  • Maintainability: {metrics['maintainability_score']:.3f}")
        print()
        
        print("📈 Technical Metrics:")
        print(f"  • Technical Debt: {metrics['technical_debt']:.2f}")
        print(f"  • Symbols Count: {insights['symbols_count']}")
        print()
        
        if args.detailed:
            # Show detailed analysis
            analysis = self.indexer.analyze_code_quality(file_path)
            
            print("🔍 Detailed Analysis:")
            print(f"  • Total Suggestions: {len(analysis.suggestions)}")
            
            # Group suggestions by type
            by_type = {}
            for suggestion in analysis.suggestions:
                if suggestion.improvement_type not in by_type:
                    by_type[suggestion.improvement_type] = []
                by_type[suggestion.improvement_type].append(suggestion)
            
            for suggestion_type, type_suggestions in by_type.items():
                print(f"  • {suggestion_type.title()}: {len(type_suggestions)} suggestions")
        
        return 0
    
    def _cmd_insights(self, args) -> int:
        """Handle the insights command."""
        file_path = args.file_path
        
        print(f"🧠 Comprehensive Code Insights for: {file_path}")
        print("=" * 60)
        
        # Get comprehensive insights
        insights = self.indexer.get_code_insights(file_path)
        
        # Display insights
        print("📊 Overview:")
        print(f"  • File Path: {insights['file_path']}")
        print(f"  • Symbols Count: {insights['symbols_count']}")
        print()
        
        # Quality metrics
        metrics = insights['quality_metrics']
        print("🎯 Quality Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  • {metric.replace('_', ' ').title()}: {value:.3f}")
            else:
                print(f"  • {metric.replace('_', ' ').title()}: {value}")
        print()
        
        # Recommendations
        if insights['recommendations']:
            print("💡 AI Recommendations:")
            for i, rec in enumerate(insights['recommendations'], 1):
                print(f"  {i}. {rec}")
        else:
            print("✅ No specific recommendations")
        print()
        
        # Symbol embeddings info
        if insights['symbol_embeddings']:
            print("🔤 Symbol Analysis:")
            print(f"  • Symbols with embeddings: {len(insights['symbol_embeddings'])}")
            print("  • Embedding dimension: 768")
            print("  • Semantic understanding: Enabled")
        
        return 0
    
    def _cmd_similar(self, args) -> int:
        """Handle the similar code command."""
        code = args.code
        context = args.context or ""
        threshold = args.threshold
        
        print(f"🔍 Finding Similar Code for:")
        print(f"Code: {code}")
        if context:
            print(f"Context: {context}")
        print(f"Similarity Threshold: {threshold}")
        print("=" * 50)
        
        # Find similar code
        similar_code = self.indexer.find_similar_code(code, context, threshold)
        
        if not similar_code:
            print("❌ No similar code found")
            print("💡 Try lowering the similarity threshold or providing more context")
            return 0
        
        print(f"✅ Found {len(similar_code)} similar code snippets:")
        print()
        
        # Display results
        for i, code_embedding in enumerate(similar_code, 1):
            print(f"🔗 Similar Code {i}:")
            print(f"  • File: {code_embedding.file_path}")
            print(f"  • Line: {code_embedding.line_number}")
            print(f"  • Symbol: {code_embedding.symbol_name}")
            print(f"  • Similarity: {code_embedding.similarity_score:.3f}")
            print(f"  • Code: {code_embedding.code[:100]}...")
            print()
        
        return 0
    
    def _cmd_export(self, args) -> int:
        """Handle the export command."""
        output_file = args.output
        
        print(f"💾 Exporting AI-Enhanced Analysis to: {output_file}")
        print("=" * 50)
        
        # Export enhanced index
        export_path = self.indexer.export_enhanced_index(output_file)
        
        print(f"✅ AI-enhanced analysis exported successfully!")
        print(f"📁 Output file: {export_path}")
        
        # Show file size
        export_file = Path(export_path)
        if export_file.exists():
            size_mb = export_file.stat().st_size / (1024 * 1024)
            print(f"📊 File size: {size_mb:.2f} MB")
        
        return 0
    
    def _cmd_status(self, args) -> int:
        """Handle the status command."""
        try:
            # Get detailed AI capabilities status
            status = self.indexer.get_ai_capabilities_status()
            
            print("🤖 AI Capabilities Status")
            print("=" * 40)
            
            # Sentence Transformers status
            st_status = status['sentence_transformers']
            if st_status['available']:
                print(f"✅ Sentence Transformers: Available")
                print(f"   • Model: {st_status['model']}")
                print(f"   • Dimension: {st_status['dimension']}")
            else:
                print("❌ Sentence Transformers: Not available")
                print("   • Install with: pip install sentence-transformers")
            
            # OpenAI status
            openai_status = status['openai']
            if openai_status['available']:
                print("✅ OpenAI GPT-4: Available")
                print("   • API Key: Configured")
                print("   • Client: Initialized")
            elif openai_status['api_key_configured']:
                print("⚠️  OpenAI GPT-4: API Key configured but client not initialized")
                print("   • Check API key validity")
            else:
                print("❌ OpenAI GPT-4: Not available")
                print("   • Set OPENAI_API_KEY environment variable")
            
            # Gemini status
            gemini_status = status['gemini']
            if gemini_status['available']:
                print("✅ Google Gemini: Available")
                print("   • API Key: Configured")
                print("   • Model: Initialized")
            elif gemini_status['api_key_configured']:
                print("⚠️  Google Gemini: API Key configured but model not initialized")
                print("   • Check API key validity")
            else:
                print("❌ Google Gemini: Not available")
                print("   • Set GEMINI_API_KEY environment variable")
            
            # Overall status
            available_count = sum([
                st_status['available'],
                openai_status['available'],
                gemini_status['available']
            ])
            
            print()
            if available_count >= 2:
                print("🎉 Full AI capabilities available!")
            elif available_count >= 1:
                print("✅ Basic AI capabilities available")
            else:
                print("⚠️  Limited AI capabilities - install dependencies")
                
        except Exception as e:
            print(f"❌ Error checking AI capabilities: {e}")
            print("   • Try running: pip install -r requirements.txt")
        
        return 0

def main():
    """Main entry point for AI-enhanced CLI."""
    cli = AICLI()
    sys.exit(cli.run(sys.argv[1:]))

if __name__ == "__main__":
    main() 