#!/usr/bin/env python3
"""
CLI interface for the Cursor-like indexing system.
Provides commands for indexing, searching, and analyzing codebases.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import json
from tabulate import tabulate

from .cursor_indexer import create_cursor_indexer, Symbol

class IndexCLI:
    """Command-line interface for the Cursor-like indexer."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.indexer = create_cursor_indexer(str(self.project_root))
    
    def run(self, args: List[str]) -> int:
        """Run the CLI with the given arguments."""
        parser = self._create_parser()
        parsed_args = parser.parse_args(args)
        
        try:
            if parsed_args.command == 'index':
                return self._cmd_index(parsed_args)
            elif parsed_args.command == 'search':
                return self._cmd_search(parsed_args)
            elif parsed_args.command == 'symbols':
                return self._cmd_symbols(parsed_args)
            elif parsed_args.command == 'deps':
                return self._cmd_deps(parsed_args)
            elif parsed_args.command == 'overview':
                return self._cmd_overview(parsed_args)
            elif parsed_args.command == 'export':
                return self._cmd_export(parsed_args)
            elif parsed_args.command == 'stats':
                return self._cmd_stats(parsed_args)
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
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description="Cursor-like codebase indexer for gcode",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Index the entire codebase
  gcode-index index
  
  # Search for functions
  gcode-index search function
  
  # Search for specific symbol
  gcode-index search "class User"
  
  # Show symbols in a file
  gcode-index symbols gcode/agent.py
  
  # Show dependencies
  gcode-index deps gcode/agent.py
  
  # Get project overview
  gcode-index overview
  
  # Export index
  gcode-index export
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Index command
        index_parser = subparsers.add_parser('index', help='Index the codebase')
        index_parser.add_argument('--force', '-f', action='store_true', 
                                help='Force re-indexing of all files')
        index_parser.add_argument('--verbose', '-v', action='store_true',
                                help='Verbose output')
        
        # Search command
        search_parser = subparsers.add_parser('search', help='Search for symbols')
        search_parser.add_argument('query', help='Search query')
        search_parser.add_argument('--kind', '-k', help='Filter by symbol kind')
        search_parser.add_argument('--file', '-f', help='Filter by file path')
        search_parser.add_argument('--limit', '-l', type=int, default=20,
                                 help='Maximum number of results')
        
        # Symbols command
        symbols_parser = subparsers.add_parser('symbols', help='Show symbols in a file')
        symbols_parser.add_argument('file_path', help='Path to the file')
        symbols_parser.add_argument('--kind', '-k', help='Filter by symbol kind')
        
        # Dependencies command
        deps_parser = subparsers.add_parser('deps', help='Show file dependencies')
        deps_parser.add_argument('file_path', help='Path to the file')
        deps_parser.add_argument('--direction', '-d', choices=['in', 'out', 'both'], 
                               default='both', help='Dependency direction')
        
        # Overview command
        overview_parser = subparsers.add_parser('overview', help='Show project overview')
        overview_parser.add_argument('--detailed', '-d', action='store_true',
                                   help='Show detailed information')
        
        # Export command
        export_parser = subparsers.add_parser('export', help='Export index data')
        export_parser.add_argument('--output', '-o', default='.gcode_index_export.json',
                                 help='Output file path')
        
        # Stats command
        stats_parser = subparsers.add_parser('stats', help='Show index statistics')
        
        return parser
    
    def _cmd_index(self, args) -> int:
        """Handle the index command."""
        print("🔍 Starting codebase indexing...")
        
        if args.verbose:
            print(f"Project root: {self.project_root}")
            print(f"Database: {self.indexer.db_path}")
        
        stats = self.indexer.index_codebase(force_reindex=args.force)
        
        print(f"✅ Indexing complete!")
        print(f"📊 Files indexed: {stats['total_files']}")
        print(f"🔤 Symbols found: {stats['total_symbols']}")
        print(f"⏱️  Time taken: {stats['last_index_time'] - stats.get('start_time', 0):.2f}s")
        
        return 0
    
    def _cmd_search(self, args) -> int:
        """Handle the search command."""
        print(f"🔍 Searching for: {args.query}")
        
        if args.kind:
            print(f"📝 Filtering by kind: {args.kind}")
        if args.file:
            print(f"📁 Filtering by file: {args.file}")
        
        symbols = self.indexer.search_symbols(args.query, args.kind, args.file)
        
        if not symbols:
            print("❌ No symbols found")
            return 0
        
        # Limit results
        symbols = symbols[:args.limit]
        
        print(f"✅ Found {len(symbols)} symbols:")
        print()
        
        # Prepare table data
        table_data = []
        for symbol in symbols:
            table_data.append([
                symbol.name,
                symbol.kind,
                symbol.file_path,
                f"{symbol.line_number}:{symbol.column}",
                symbol.visibility,
                symbol.complexity,
                symbol.signature[:50] + "..." if len(symbol.signature) > 50 else symbol.signature
            ])
        
        headers = ["Name", "Kind", "File", "Location", "Visibility", "Complexity", "Signature"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        if len(symbols) == args.limit:
            print(f"\n... and {len(self.indexer.search_symbols(args.query, args.kind, args.file)) - args.limit} more results")
        
        return 0
    
    def _cmd_symbols(self, args) -> int:
        """Handle the symbols command."""
        file_path = args.file_path
        
        print(f"📁 Symbols in: {file_path}")
        
        if args.kind:
            print(f"📝 Filtering by kind: {args.kind}")
        
        # Try both relative and absolute paths
        symbols = self.indexer.get_file_symbols(file_path)
        if not symbols:
            # Try with absolute path
            abs_path = str(Path(file_path).resolve())
            symbols = self.indexer.get_file_symbols(abs_path)
        if not symbols:
            # Try searching by filename
            symbols = self.indexer.search_symbols("", file_path=Path(file_path).name)
        
        if args.kind:
            symbols = [s for s in symbols if s.kind == args.kind]
        
        if not symbols:
            print("❌ No symbols found in this file")
            return 0
        
        print(f"✅ Found {len(symbols)} symbols:")
        print()
        
        # Group symbols by kind
        symbols_by_kind = {}
        for symbol in symbols:
            if symbol.kind not in symbols_by_kind:
                symbols_by_kind[symbol.kind] = []
            symbols_by_kind[symbol.kind].append(symbol)
        
        for kind, kind_symbols in symbols_by_kind.items():
            print(f"📝 {kind.title()}s ({len(kind_symbols)}):")
            
            table_data = []
            for symbol in kind_symbols:
                table_data.append([
                    symbol.name,
                    f"{symbol.line_number}:{symbol.column}",
                    symbol.visibility,
                    symbol.complexity,
                    symbol.signature[:60] + "..." if len(symbol.signature) > 60 else symbol.signature
                ])
            
            headers = ["Name", "Location", "Visibility", "Complexity", "Signature"]
            print(tabulate(table_data, headers=headers, tablefmt="simple"))
            print()
        
        return 0
    
    def _cmd_deps(self, args) -> int:
        """Handle the dependencies command."""
        file_path = args.file_path
        
        print(f"🔗 Dependencies for: {file_path}")
        print()
        
        if args.direction in ['in', 'both']:
            print("📥 Dependencies (imports):")
            deps = self.indexer.get_dependencies(file_path)
            if deps:
                for dep in deps:
                    print(f"  • {dep}")
            else:
                print("  No dependencies found")
            print()
        
        if args.direction in ['out', 'both']:
            print("📤 Dependents (imported by):")
            dependents = self.indexer.get_dependents(file_path)
            if dependents:
                for dep in dependents:
                    print(f"  • {dep}")
            else:
                print("  No dependents found")
            print()
        
        return 0
    
    def _cmd_overview(self, args) -> int:
        """Handle the overview command."""
        print("🏗️  Project Overview")
        print("=" * 50)
        
        overview = self.indexer.get_project_overview()
        
        # Basic stats
        stats = overview['index_stats']
        print(f"📊 Index Statistics:")
        print(f"  • Total files: {stats['total_files']}")
        print(f"  • Total symbols: {stats['total_symbols']}")
        print(f"  • Index version: {stats['index_version']}")
        print()
        
        # File statistics
        file_stats = overview['file_statistics']
        if file_stats:
            print("📁 File Statistics:")
            table_data = []
            for file_type, info in file_stats.items():
                table_data.append([
                    file_type,
                    info['count'],
                    info['complexity'] or 0
                ])
            
            headers = ["Type", "Count", "Total Complexity"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print()
        
        # Symbol statistics
        symbol_stats = overview['symbol_statistics']
        if symbol_stats:
            print("🔤 Symbol Statistics:")
            table_data = []
            for kind, info in symbol_stats.items():
                table_data.append([
                    kind,
                    info['count'],
                    f"{info['avg_complexity']:.1f}" if info['avg_complexity'] else "0.0"
                ])
            
            headers = ["Kind", "Count", "Avg Complexity"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print()
        
        if args.detailed:
            # Most complex files
            complex_files = overview['most_complex_files']
            if complex_files:
                print("🔥 Most Complex Files:")
                table_data = []
                for file_info in complex_files[:10]:
                    table_data.append([
                        file_info['path'],
                        file_info['complexity']
                    ])
                
                headers = ["File", "Complexity"]
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
                print()
            
            # Most complex symbols
            complex_symbols = overview['most_complex_symbols']
            if complex_symbols:
                print("🔥 Most Complex Symbols:")
                table_data = []
                for symbol_info in complex_symbols[:10]:
                    table_data.append([
                        symbol_info['name'],
                        symbol_info['kind'],
                        symbol_info['file'],
                        symbol_info['complexity']
                    ])
                
                headers = ["Name", "Kind", "File", "Complexity"]
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
                print()
        
        return 0
    
    def _cmd_export(self, args) -> int:
        """Handle the export command."""
        print(f"💾 Exporting index to: {args.output}")
        
        output_path = self.indexer.export_index(args.output)
        
        print(f"✅ Index exported successfully!")
        print(f"📁 Output file: {output_path}")
        
        return 0
    
    def _cmd_stats(self, args) -> int:
        """Handle the stats command."""
        stats = self.indexer._get_index_stats()
        
        print("📊 Index Statistics")
        print("=" * 30)
        print(f"Total files: {stats['total_files']}")
        print(f"Total symbols: {stats['total_symbols']}")
        print(f"Index version: {stats['index_version']}")
        print(f"Last index time: {stats['last_index_time']}")
        
        return 0

def main():
    """Main entry point."""
    cli = IndexCLI()
    sys.exit(cli.run(sys.argv[1:]))

if __name__ == "__main__":
    main() 