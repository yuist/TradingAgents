#!/usr/bin/env python3
"""
TradingAgents CLI Launcher
This script provides a simple way to run the analyze command.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli.main import analyze

if __name__ == "__main__":
    analyze()