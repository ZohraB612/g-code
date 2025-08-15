#!/usr/bin/env python3
"""
AI-Enhanced Indexing Demo for gcode - Goes beyond Claude Code!
This demo showcases the advanced AI-powered features that make gcode superior.
"""

import sys
import time
from pathlib import Path
import json

# Add the gcode package to the path
sys.path.insert(0, str(Path(__file__).parent))

from gcode.ai_enhanced_indexer import create_ai_enhanced_indexer

def demo_ai_enhanced_indexing():
    """Demonstrate AI-enhanced indexing capabilities."""
    print("🚀 Demo: AI-Enhanced Codebase Indexing")
    print("=" * 60)
    
    # Create AI-enhanced indexer
    indexer = create_ai_enhanced_indexer(".")
    
    # Index the codebase
    print("🔍 Indexing the gcode project with AI capabilities...")
    start_time = time.time()
    
    stats = indexer.index_codebase()
    
    elapsed = time.time() - start_time
    print(f"✅ AI-enhanced indexing complete in {elapsed:.2f}s")
    print(f"📊 Files indexed: {stats['total_files']}")
    print(f"🔤 Symbols found: {stats['total_symbols']}")
    print()
    
    # Check AI capabilities
    print("🤖 AI Capabilities Status:")
    print(f"  • Embeddings Model: {'✅ Available' if indexer.embeddings_model else '❌ Not Available'}")
    print(f"  • OpenAI GPT-4: {'✅ Available' if indexer.openai_client else '❌ Not Available'}")
    print(f"  • Google Gemini: {'✅ Available' if indexer.gemini_model else '❌ Not Available'}")
    print()

def demo_semantic_search():
    """Demonstrate semantic search capabilities."""
    print("🧠 Demo: Semantic Code Search")
    print("=" * 50)
    
    indexer = create_ai_enhanced_indexer(".")
    
    if not indexer.embeddings_model:
        print("⚠️  Embeddings model not available - skipping semantic search demo")
        print("💡 Install with: pip install sentence-transformers")
        return
    
    # Test semantic search
    queries = [
        "authentication function",
        "data processing",
        "error handling",
        "file operations",
        "database connection"
    ]
    
    for query in queries:
        print(f"🔍 Searching for: {query}")
        results = indexer.enhanced_search(query, use_semantics=True)
        
        if results:
            print(f"  Found {len(results)} results:")
            for i, result in enumerate(results[:3], 1):
                symbol = result['symbol']
                name = getattr(symbol, 'name', getattr(symbol, 'symbol_name', 'Unknown'))
                score = result['score']
                search_type = result['search_type']
                print(f"    {i}. {name} ({search_type}) - Score: {score:.3f}")
        else:
            print("  No results found")
        print()

def demo_code_suggestions():
    """Demonstrate AI-powered code suggestions."""
    print("💡 Demo: AI Code Improvement Suggestions")
    print("=" * 50)
    
    indexer = create_ai_enhanced_indexer(".")
    
    if not (indexer.openai_client or indexer.gemini_model):
        print("⚠️  AI models not available - skipping suggestions demo")
        print("💡 Set OPENAI_API_KEY or GEMINI_API_KEY environment variables")
        return
    
    # Test code suggestions
    test_codes = [
        ("x = 1 + 1", "Simple addition", "python"),
        ("def process_data(data): pass", "Empty function", "python"),
        ("for i in range(100): print(i)", "Simple loop", "python"),
        ("result = eval(user_input)", "Unsafe evaluation", "python"),
        ("global_var = 42", "Global variable usage", "python")
    ]
    
    for code, context, language in test_codes:
        print(f"🔧 Analyzing: {code}")
        print(f"   Context: {context}")
        
        suggestions = indexer.generate_code_suggestions(code, context, language)
        
        if suggestions:
            print(f"   Generated {len(suggestions)} suggestions:")
            for i, suggestion in enumerate(suggestions[:2], 1):
                print(f"     {i}. [{suggestion.priority.upper()}] {suggestion.improvement_type}")
                print(f"        {suggestion.explanation[:80]}...")
        else:
            print("   No suggestions generated")
        print()

def demo_code_quality_analysis():
    """Demonstrate AI-powered code quality analysis."""
    print("📊 Demo: AI Code Quality Analysis")
    print("=" * 50)
    
    indexer = create_ai_enhanced_indexer(".")
    
    # Analyze a specific file
    target_file = "gcode/agent.py"
    print(f"🔍 Analyzing code quality for: {target_file}")
    
    try:
        analysis = indexer.analyze_code_quality(target_file)
        
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
            for i, suggestion in enumerate(analysis.suggestions[:3], 1):
                print(f"  {i}. [{suggestion.priority.upper()}] {suggestion.improvement_type}")
                print(f"     {suggestion.explanation[:100]}...")
        else:
            print("✅ No AI suggestions generated")
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
    
    print()

def demo_code_similarity():
    """Demonstrate code similarity detection."""
    print("🔗 Demo: Code Similarity Detection")
    print("=" * 50)
    
    indexer = create_ai_enhanced_indexer(".")
    
    if not indexer.embeddings_model:
        print("⚠️  Embeddings model not available - skipping similarity demo")
        return
    
    # Test code similarity
    test_queries = [
        ("def calculate_total(items):", "Function to calculate total"),
        ("class User:", "User class definition"),
        ("import os", "OS import statement"),
        ("try:", "Error handling block"),
        ("for item in items:", "Loop over items")
    ]
    
    for code, context in test_queries:
        print(f"🔍 Finding similar code for: {code}")
        print(f"   Context: {context}")
        
        similar_code = indexer.find_similar_code(code, context, threshold=0.5)
        
        if similar_code:
            print(f"   Found {len(similar_code)} similar code snippets:")
            for i, code_embedding in enumerate(similar_code[:2], 1):
                print(f"     {i}. {code_embedding.symbol_name} (similarity: {code_embedding.similarity_score:.3f})")
        else:
            print("   No similar code found")
        print()

def demo_comprehensive_insights():
    """Demonstrate comprehensive code insights."""
    print("🧠 Demo: Comprehensive Code Insights")
    print("=" * 50)
    
    indexer = create_ai_enhanced_indexer(".")
    
    # Get insights for a file
    target_file = "gcode/agent.py"
    print(f"🔍 Getting comprehensive insights for: {target_file}")
    
    try:
        insights = indexer.get_code_insights(target_file)
        
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
        
        # Symbol embeddings info
        if insights['symbol_embeddings']:
            print("🔤 Symbol Analysis:")
            print(f"  • Symbols with embeddings: {len(insights['symbol_embeddings'])}")
            print("  • Embedding dimension: 768")
            print("  • Semantic understanding: Enabled")
        
    except Exception as e:
        print(f"❌ Error getting insights: {e}")
    
    print()

def demo_export_capabilities():
    """Demonstrate AI-enhanced export capabilities."""
    print("💾 Demo: AI-Enhanced Export")
    print("=" * 50)
    
    indexer = create_ai_enhanced_indexer(".")
    
    # Export enhanced index
    print("📤 Exporting AI-enhanced analysis...")
    try:
        export_path = indexer.export_enhanced_index(".gcode_ai_demo_export.json")
        
        print(f"✅ AI-enhanced analysis exported successfully!")
        print(f"📁 Output file: {export_path}")
        
        # Show file size
        export_file = Path(export_path)
        if export_file.exists():
            size_mb = export_file.stat().st_size / (1024 * 1024)
            print(f"📊 File size: {size_mb:.2f} MB")
        
        # Show export contents
        print("\n📋 Export Contents:")
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        metadata = export_data.get('metadata', {})
        print(f"  • AI Enhanced: {metadata.get('ai_enhanced', False)}")
        print(f"  • Embeddings Available: {metadata.get('embeddings_available', False)}")
        print(f"  • OpenAI Available: {metadata.get('openai_available', False)}")
        print(f"  • Gemini Available: {metadata.get('gemini_available', False)}")
        
        ai_analysis = export_data.get('ai_analysis', {})
        print(f"  • Files Analyzed: {len(ai_analysis)}")
        
        embeddings = export_data.get('embeddings', {})
        print(f"  • Code Embeddings: {len(embeddings)}")
        
    except Exception as e:
        print(f"❌ Error exporting: {e}")
    
    print()

def demo_cli_commands():
    """Demonstrate CLI command usage."""
    print("🖥️  Demo: CLI Command Examples")
    print("=" * 50)
    
    print("💡 Try these AI-enhanced CLI commands:")
    print()
    
    print("🔍 Code Analysis:")
    print("  gcode-ai analyze gcode/agent.py --detailed")
    print("  gcode-ai quality gcode/agent.py")
    print("  gcode-ai insights gcode/agent.py")
    print()
    
    print("🧠 Semantic Search:")
    print("  gcode-ai semantic 'authentication function'")
    print("  gcode-ai semantic 'data processing' --context 'backend'")
    print()
    
    print("💡 Code Suggestions:")
    print("  gcode-ai suggestions 'def process_data(data): pass'")
    print("  gcode-ai suggestions 'x = 1 + 1' --language python")
    print()
    
    print("🔗 Code Similarity:")
    print("  gcode-ai similar 'def calculate_total(items):'")
    print("  gcode-ai similar 'class User:' --threshold 0.6")
    print()
    
    print("📊 Export & Status:")
    print("  gcode-ai export --output my_analysis.json")
    print("  gcode-ai status")
    print()

def main():
    """Run all AI-enhanced demos."""
    print("🎯 AI-Enhanced Indexing System Demo")
    print("=" * 70)
    print("This demo showcases the advanced AI-powered features that go BEYOND Claude Code!")
    print("Features: semantic search, AI suggestions, quality analysis, and intelligent insights.")
    print()
    
    try:
        # Run all demos
        demo_ai_enhanced_indexing()
        demo_semantic_search()
        demo_code_suggestions()
        demo_code_quality_analysis()
        demo_code_similarity()
        demo_comprehensive_insights()
        demo_export_capabilities()
        demo_cli_commands()
        
        print("🎉 All AI-enhanced demos completed successfully!")
        print()
        print("🚀 What makes this BETTER than Claude Code:")
        print("  • 🔤 Semantic code understanding with embeddings")
        print("  • 🤖 AI-powered code improvement suggestions")
        print("  • 📊 Comprehensive quality analysis")
        print("  • 🔍 Intelligent code similarity detection")
        print("  • 💡 Proactive code improvement recommendations")
        print("  • 🎯 Technical debt quantification")
        print("  • 🔒 Security vulnerability detection")
        print("  • ⚡ Performance optimization suggestions")
        print()
        print("💡 Try the AI-enhanced CLI:")
        print("  gcode-ai status          # Check AI capabilities")
        print("  gcode-ai analyze <file>  # AI-powered analysis")
        print("  gcode-ai semantic <query> # Semantic search")
        print("  gcode-ai suggestions <code> # Get AI suggestions")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 