#!/usr/bin/env python3
"""
Simple test script to demonstrate working collapsible sections.
"""

import sys
sys.path.append('gcode')

from gcode.agent import ProfessionalUI

def test_collapsible_sections():
    """Test the collapsible sections functionality."""
    print("🧪 Testing Collapsible Sections")
    print("=" * 40)
    
    ui = ProfessionalUI()
    
    # Create some sections
    ui.section("Project Overview", 
               "This project contains multiple Python files and tools for code analysis.", 
               collapsible=True, expanded=False)
    
    ui.section("Available Tools", 
               "• read_file\n• write_file\n• analyze_python_file\n• git_status", 
               collapsible=True, expanded=True)
    
    ui.section("Recent Activity", 
               "Last 3 actions:\n1. Created test file\n2. Analyzed code\n3. Generated tests", 
               collapsible=True, expanded=False)
    
    ui.section("Non-collapsible", 
               "This section cannot be collapsed.", 
               collapsible=False, expanded=True)
    
    print("Initial state:")
    print("=" * 20)
    for i in range(1, 5):
        print(ui.render_section(i))
    
    print("\n" + "=" * 40)
    print("After toggling sections 1 and 3:")
    print("=" * 20)
    
    # Toggle sections 1 and 3
    ui.toggle_section(1)  # Expand section 1
    ui.toggle_section(3)  # Expand section 3
    
    for i in range(1, 5):
        print(ui.render_section(i))
    
    print("\n" + "=" * 40)
    print("Interactive mode - you can now toggle sections:")
    print("Commands: '1' to toggle section 1, '2' to toggle section 2, etc.")
    print("Type 'q' to quit")
    
    while True:
        try:
            cmd = input("\nCommand (1-4, or 'q'): ").strip()
            if cmd.lower() == 'q':
                break
            elif cmd.isdigit():
                section_num = int(cmd)
                if 1 <= section_num <= 4:
                    if ui.sections[section_num]['collapsible']:
                        ui.toggle_section(section_num)
                        print(f"Toggled section {section_num}")
                        
                        # Re-render all sections
                        print("\nCurrent state:")
                        for i in range(1, 5):
                            print(ui.render_section(i))
                    else:
                        print(f"Section {section_num} is not collapsible")
                else:
                    print("Invalid section number")
            else:
                print("Invalid command")
        except KeyboardInterrupt:
            break
    
    print("\nDemo completed!")

if __name__ == "__main__":
    test_collapsible_sections()
