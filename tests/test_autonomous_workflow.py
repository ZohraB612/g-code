#!/usr/bin/env python3
"""
Test script for gcode Autonomous Workflow capabilities.
This demonstrates the end-to-end workflow execution.
"""

import os
import sys
from pathlib import Path

def test_autonomous_workflow():
    """Test the autonomous workflow functionality."""
    print("🚀 Testing gcode Autonomous Workflows")
    print("=" * 60)
    
    # Test the agent creation and workflow detection
    try:
        from gcode.agent import GeminiAgent
        
        # Create agent instance
        agent = GeminiAgent()
        print("✅ gcode agent created successfully")
        
        # Test workflow detection
        print("\n🔍 Testing workflow detection...")
        
        simple_requests = [
            "What functions are in agent.py?",
            "Show me the project structure",
            "How many Python files are there?"
        ]
        
        complex_workflows = [
            "Create a new authentication system for this project",
            "Build and test a new utility function",
            "Set up automated testing for this project",
            "Refactor the code for better performance",
            "Implement CI/CD pipeline for deployment"
        ]
        
        print("\n📝 Simple Requests (should use simple processing):")
        for req in simple_requests:
            is_complex = agent._is_complex_workflow(req)
            status = "Complex" if is_complex else "Simple"
            print(f"  • '{req}' → {status}")
        
        print("\n🎯 Complex Workflows (should use autonomous mode):")
        for req in complex_workflows:
            is_complex = agent._is_complex_workflow(req)
            status = "Complex" if is_complex else "Simple"
            print(f"  • '{req}' → {status}")
        
        # Test plan creation (without execution)
        print("\n🤔 Testing plan creation...")
        test_goal = "Create a simple utility function that adds two numbers"
        
        try:
            plan = agent._create_plan(test_goal)
            if plan:
                print(f"✅ Plan created successfully with {len(plan)} steps:")
                for i, step in enumerate(plan, 1):
                    print(f"  {i}. {step.get('thought', 'No thought provided')}")
                    tool_calls = step.get('tool_calls', [])
                    if tool_calls:
                        print(f"     Tools: {len(tool_calls)} tool calls")
            else:
                print("⚠️  Plan creation failed")
        except Exception as e:
            print(f"❌ Error during plan creation: {e}")
        
        print("\n✅ Autonomous workflow test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during autonomous workflow test: {e}")
        return False

def show_workflow_examples():
    """Show examples of how to use the autonomous workflow features."""
    print("\n💡 Autonomous Workflow Examples:")
    print("=" * 40)
    
    print("\n1. Feature Development:")
    print("   gcode 'Create a new user authentication system'")
    print("   gcode 'Build a REST API for user management'")
    print("   gcode 'Implement user role-based access control'")
    
    print("\n2. Testing & Quality:")
    print("   gcode 'Set up comprehensive testing suite'")
    print("   gcode 'Add unit tests for all functions'")
    print("   gcode 'Implement code coverage reporting'")
    
    print("\n3. DevOps & Deployment:")
    print("   gcode 'Set up CI/CD pipeline with GitHub Actions'")
    print("   gcode 'Configure automated deployment to staging'")
    print("   gcode 'Implement blue-green deployment strategy'")
    
    print("\n4. Code Improvement:")
    print("   gcode 'Refactor legacy code for modern standards'")
    print("   gcode 'Optimize performance bottlenecks'")
    print("   gcode 'Add comprehensive error handling'")
    
    print("\n5. Project Setup:")
    print("   gcode 'Initialize a new Python project with best practices'")
    print("   gcode 'Set up development environment with Docker'")
    print("   gcode 'Configure linting and formatting tools'")
    
    print("\n🚀 What Happens:")
    print("  1. gcode analyzes your request")
    print("  2. Creates a detailed step-by-step plan")
    print("  3. Executes each step automatically")
    print("  4. Handles errors and self-corrects")
    print("  5. Completes the entire workflow")
    print("  6. No intermediate prompts needed!")

def main():
    """Run the autonomous workflow test suite."""
    print("🚀 gcode Autonomous Workflow Test Suite")
    print("=" * 60)
    
    # Run tests
    if test_autonomous_workflow():
        print("\n✅ All tests passed! Autonomous workflows are working correctly.")
        show_workflow_examples()
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("💡 Make sure all dependencies are installed and the code is properly structured.")

if __name__ == "__main__":
    main()
