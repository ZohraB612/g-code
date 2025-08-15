#!/usr/bin/env python3
"""
Test script for gcode Deep Codebase Understanding.
This demonstrates the "agentic search" capabilities that make Claude Code powerful.
"""

import os
import sys
from pathlib import Path

def test_deep_understanding():
    """Test the deep codebase understanding functionality."""
    print("🧠 Testing gcode Deep Codebase Understanding")
    print("=" * 60)
    
    # Test the analyzer module
    try:
        from gcode.analyzer import create_analyzer
        print("✅ CodebaseAnalyzer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import analyzer: {e}")
        return False
    
    # Test the tools module
    try:
        from gcode.tools import query_codebase, deep_codebase_analysis
        print("✅ Deep understanding tools imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import tools: {e}")
        return False
    
    # Test the agent integration
    try:
        from gcode.agent import GeminiAgent
        print("✅ gcode agent imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import agent: {e}")
        return False
    
    # Test basic analyzer functionality
    try:
        print("\n🔍 Testing CodebaseAnalyzer...")
        analyzer = create_analyzer(".")
        
        # Test file analysis
        test_file = Path("gcode/agent.py")
        if test_file.exists():
            analyzer.analyze_file(test_file, "python")
            print(f"✅ File analysis: {test_file} analyzed")
        else:
            print(f"⚠️  Test file not found: {test_file}")
        
        # Test knowledge graph creation
        print("\n🧠 Building knowledge graph...")
        knowledge_graph = analyzer.analyze()
        print(f"✅ Knowledge graph built: {len(knowledge_graph)} files")
        
        # Test querying
        print("\n🔍 Testing knowledge graph queries...")
        if knowledge_graph:
            # Test architecture query
            arch_result = analyzer.query("show me the architecture")
            print(f"✅ Architecture query: {arch_result[:100]}...")
            
            # Test framework query
            framework_result = analyzer.query("what frameworks are used")
            print(f"✅ Framework query: {framework_result[:100]}...")
        
    except Exception as e:
        print(f"❌ Error during analyzer test: {e}")
        return False
    
    # Test tools integration
    try:
        print("\n🛠️  Testing deep understanding tools...")
        
        # Test deep analysis
        analysis_result = deep_codebase_analysis()
        print(f"✅ Deep analysis: {analysis_result[:100]}...")
        
        # Test codebase querying
        query_result = query_codebase("what functions are in agent.py")
        print(f"✅ Codebase query: {query_result[:100]}...")
        
    except Exception as e:
        print(f"❌ Error during tools test: {e}")
        return False
    
    # Test agent integration
    try:
        print("\n🤖 Testing agent integration...")
        
        # Create agent instance
        agent = GeminiAgent()
        print("✅ Agent created successfully")
        
        # Test knowledge graph loading
        if hasattr(agent.context, 'knowledge_graph'):
            print(f"✅ Knowledge graph loaded: {len(agent.context.knowledge_graph)} files")
        else:
            print("⚠️  Knowledge graph not loaded in context")
        
        # Test query method
        if hasattr(agent, 'query_codebase'):
            print("✅ query_codebase method available")
        else:
            print("⚠️  query_codebase method not found")
        
    except Exception as e:
        print(f"❌ Error during agent integration test: {e}")
        return False
    
    print("\n🎉 Deep understanding test completed successfully!")
    return True

def show_usage_examples():
    """Show examples of how to use the deep understanding features."""
    print("\n💡 Usage Examples:")
    print("=" * 40)
    
    print("\n1. Start gcode and ask questions:")
    print("   gcode 'query what functions are in agent.py?'")
    print("   gcode 'query show me the architecture'")
    print("   gcode 'query what frameworks are used?'")
    
    print("\n2. Use interactive mode:")
    print("   gcode")
    print("   > query find files with authentication")
    print("   > query complexity analysis")
    print("   > query test coverage")
    
    print("\n3. Force reanalysis:")
    print("   gcode 'analyze'")
    
    print("\n4. Analyze specific file:")
    print("   gcode 'analyze gcode/agent.py'")
    
    print("\n🚀 This gives you Claude Code-level understanding!")

def main():
    """Run the deep understanding test suite."""
    print("🚀 gcode Deep Codebase Understanding Test Suite")
    print("=" * 60)
    
    # Run tests
    if test_deep_understanding():
        print("\n✅ All tests passed! Deep understanding is working correctly.")
        show_usage_examples()
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("💡 Make sure all dependencies are installed and the code is properly structured.")

if __name__ == "__main__":
    main()
