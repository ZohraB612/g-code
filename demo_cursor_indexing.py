#!/usr/bin/env python3
"""
Demo script for the Cursor-like indexing system in gcode.
This demonstrates the advanced codebase understanding capabilities.
"""

import sys
import time
from pathlib import Path

# Add the gcode package to the path
sys.path.insert(0, str(Path(__file__).parent))

from gcode.cursor_indexer import create_cursor_indexer

def demo_basic_indexing():
    """Demonstrate basic indexing capabilities."""
    print("🚀 Demo: Basic Codebase Indexing")
    print("=" * 50)
    
    # Create indexer
    indexer = create_cursor_indexer(".")
    
    # Index the codebase
    print("🔍 Indexing the gcode project...")
    start_time = time.time()
    
    stats = indexer.index_codebase()
    
    elapsed = time.time() - start_time
    print(f"✅ Indexing complete in {elapsed:.2f}s")
    print(f"📊 Files indexed: {stats['total_files']}")
    print(f"🔤 Symbols found: {stats['total_symbols']}")
    print()

def demo_symbol_search():
    """Demonstrate symbol search capabilities."""
    print("🔍 Demo: Symbol Search")
    print("=" * 50)
    
    indexer = create_cursor_indexer(".")
    
    # Search for functions
    print("📝 Searching for functions...")
    functions = indexer.search_symbols("function", kind="function", limit=10)
    
    if functions:
        print(f"Found {len(functions)} functions:")
        for func in functions[:5]:  # Show first 5
            print(f"  • {func.name} in {func.file_path}:{func.line_number}")
        if len(functions) > 5:
            print(f"  ... and {len(functions) - 5} more")
    print()
    
    # Search for classes
    print("🏗️  Searching for classes...")
    classes = indexer.search_symbols("class", kind="class", limit=10)
    
    if classes:
        print(f"Found {len(classes)} classes:")
        for cls in classes[:5]:  # Show first 5
            print(f"  • {cls.name} in {cls.file_path}:{cls.line_number}")
        if len(classes) > 5:
            print(f"  ... and {len(classes) - 5} more")
    print()

def demo_file_analysis():
    """Demonstrate file analysis capabilities."""
    print("📁 Demo: File Analysis")
    print("=" * 50)
    
    indexer = create_cursor_indexer(".")
    
    # Analyze a specific file
    target_file = "gcode/agent.py"
    print(f"🔍 Analyzing symbols in {target_file}...")
    
    symbols = indexer.get_file_symbols(target_file)
    
    if symbols:
        # Group by kind
        by_kind = {}
        for symbol in symbols:
            if symbol.kind not in by_kind:
                by_kind[symbol.kind] = []
            by_kind[symbol.kind].append(symbol)
        
        print(f"Found {len(symbols)} symbols:")
        for kind, kind_symbols in by_kind.items():
            print(f"  📝 {kind.title()}s: {len(kind_symbols)}")
            for symbol in kind_symbols[:3]:  # Show first 3 of each kind
                print(f"    • {symbol.name} (line {symbol.line_number})")
            if len(kind_symbols) > 3:
                print(f"    ... and {len(kind_symbols) - 3} more")
    print()

def demo_dependency_analysis():
    """Demonstrate dependency analysis capabilities."""
    print("🔗 Demo: Dependency Analysis")
    print("=" * 50)
    
    indexer = create_cursor_indexer(".")
    
    # Show dependencies for a file
    target_file = "gcode/agent.py"
    print(f"📥 Dependencies of {target_file}:")
    
    deps = indexer.get_dependencies(target_file)
    if deps:
        for dep in deps:
            print(f"  • {dep}")
    else:
        print("  No dependencies found")
    print()
    
    # Show dependents
    print(f"📤 Files that depend on {target_file}:")
    dependents = indexer.get_dependents(target_file)
    if dependents:
        for dep in dependents:
            print(f"  • {dep}")
    else:
        print("  No dependents found")
    print()

def demo_semantic_search():
    """Demonstrate semantic search capabilities."""
    print("🧠 Demo: Semantic Search")
    print("=" * 50)
    
    indexer = create_cursor_indexer(".")
    
    # Search for authentication-related code
    print("🔐 Searching for authentication-related code...")
    auth_symbols = indexer.semantic_search("authentication", context="security")
    
    if auth_symbols:
        print(f"Found {len(auth_symbols)} relevant symbols:")
        for symbol in auth_symbols[:5]:
            print(f"  • {symbol.name} ({symbol.kind}) in {symbol.file_path}")
            if hasattr(symbol, 'score'):
                print(f"    Relevance score: {symbol.score}")
    else:
        print("No authentication-related code found")
    print()
    
    # Search for testing code
    print("🧪 Searching for testing-related code...")
    test_symbols = indexer.semantic_search("test", context="testing")
    
    if test_symbols:
        print(f"Found {len(test_symbols)} relevant symbols:")
        for symbol in test_symbols[:5]:
            print(f"  • {symbol.name} ({symbol.kind}) in {symbol.file_path}")
            if hasattr(symbol, 'score'):
                print(f"    Relevance score: {symbol.score}")
    else:
        print("No testing-related code found")
    print()

def demo_project_overview():
    """Demonstrate project overview capabilities."""
    print("🏗️  Demo: Project Overview")
    print("=" * 50)
    
    indexer = create_cursor_indexer(".")
    
    # Get project overview
    overview = indexer.get_project_overview()
    
    print("📊 Project Statistics:")
    stats = overview['index_stats']
    print(f"  • Total files: {stats['total_files']}")
    print(f"  • Total symbols: {stats['total_symbols']}")
    print(f"  • Index version: {stats['index_version']}")
    print()
    
    # File type breakdown
    if overview['file_statistics']:
        print("📁 Files by Type:")
        for file_type, info in overview['file_statistics'].items():
            print(f"  • {file_type}: {info['count']} files")
        print()
    
    # Symbol breakdown
    if overview['symbol_statistics']:
        print("🔤 Symbols by Kind:")
        for kind, info in overview['symbol_statistics'].items():
            print(f"  • {kind}: {info['count']} symbols")
        print()
    
    # Most complex files
    if overview['most_complex_files']:
        print("🔥 Top 5 Most Complex Files:")
        for file_info in overview['most_complex_files'][:5]:
            print(f"  • {file_info['path']}: complexity {file_info['complexity']}")
        print()

def demo_export_capabilities():
    """Demonstrate export capabilities."""
    print("💾 Demo: Export Capabilities")
    print("=" * 50)
    
    indexer = create_cursor_indexer(".")
    
    # Export the index
    print("📤 Exporting index data...")
    export_path = indexer.export_index(".gcode_demo_export.json")
    
    print(f"✅ Index exported to: {export_path}")
    
    # Show file size
    export_file = Path(export_path)
    if export_file.exists():
        size_mb = export_file.stat().st_size / (1024 * 1024)
        print(f"📁 Export file size: {size_mb:.2f} MB")
    
    print()

def main():
    """Run all demos."""
    print("🎯 Cursor-like Indexing System Demo")
    print("=" * 60)
    print("This demo showcases the advanced codebase understanding capabilities")
    print("similar to Cursor's intelligent indexing system.")
    print()
    
    try:
        # Run all demos
        demo_basic_indexing()
        demo_symbol_search()
        demo_file_analysis()
        demo_dependency_analysis()
        demo_semantic_search()
        demo_project_overview()
        demo_export_capabilities()
        
        print("🎉 All demos completed successfully!")
        print()
        print("💡 Try using the CLI commands:")
        print("  gcode-index index          # Index the codebase")
        print("  gcode-index search function # Search for functions")
        print("  gcode-index overview       # Get project overview")
        print("  gcode-index symbols gcode/agent.py # Show file symbols")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 