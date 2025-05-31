#!/usr/bin/env python3
"""
Setup and run script for Gene Expression Explorer
Hackathon Prototype - Local Development
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def install_requirements():
    """Install required packages"""
    requirements = [
        "fastapi",
        "uvicorn[standard]",
        "pandas",
        "numpy",
        "scikit-learn",
        "plotly",
        "scipy",
        "pydantic"
    ]
    
    print("🔧 Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    return True

def setup_project_structure():
    """Create necessary directories and files"""
    print("📁 Setting up project structure...")
    
    # Create data directory
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    print("✅ Created data directory")
    
    # Create static directory for HTML
    static_dir = Path("./static")
    static_dir.mkdir(exist_ok=True)
    print("✅ Created static directory")
    
    return True

def check_files_exist():
    """Check if required files exist"""
    required_files = ["main.py", "static/dash.html"]
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def start_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting Gene Expression Explorer API...")
    print("📡 Server will be available at: http://localhost:8000")
    print("🌐 Dashboard will be available at: http://localhost:8000/dashboard")
    print("\n⏹️  Press Ctrl+C to stop the server")
    
    try:
        # Start uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def main():
    """Main setup and run function"""
    print("🧬 Gene Expression Explorer - Hackathon Setup")
    print("=" * 50)
    
    # Check if files exist
    if not check_files_exist():
        print("\n❌ Setup failed: Missing required files")
        print("Please ensure main.py and dash.html are in the current directory")
        return
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed: Could not install required packages")
        return
    
    # Setup project structure
    if not setup_project_structure():
        print("\n❌ Setup failed: Could not create project structure")
        return
    
    print("\n✅ Setup completed successfully!")
    print("\n" + "=" * 50)
    
    # Ask user if they want to start the server
    response = input("🚀 Start the server now? (y/n): ").lower().strip()
    if response in ['y', 'yes', '']:
        start_server()
    else:
        print("\n📝 To start the server manually, run:")
        print("   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        print("\n🌐 Then open: http://localhost:8000/dashboard")

if __name__ == "__main__":
    main()