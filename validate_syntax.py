#!/usr/bin/env python3
"""
Validate that the fixed code has correct syntax and imports
"""

import ast
import sys

def validate_file(filename):
    """Validate Python file syntax"""
    try:
        with open(filename, 'r') as f:
            source = f.read()
        
        # Parse the AST
        tree = ast.parse(source, filename)
        
        # Check for basic issues
        issues = []
        
        # Check for function definitions
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        print(f"Found {len(functions)} functions in {filename}")
        
        # Check for class definitions
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        print(f"Found {len(classes)} classes in {filename}")
        
        # Check for imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        print(f"Found {len(imports)} imports in {filename}")
        
        print(f"✅ {filename} syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in {filename}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating {filename}: {e}")
        return False

def main():
    """Validate all Python files"""
    files_to_check = ['CC.py', 'main.py', 'test_pipeline.py']
    
    all_valid = True
    for filename in files_to_check:
        if not validate_file(filename):
            all_valid = False
        print()
    
    if all_valid:
        print("🎉 All files have valid syntax!")
        return True
    else:
        print("❌ Some files have syntax errors")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
