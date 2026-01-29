#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def setup_python_path():
    """Add project root to Python path"""
    project_root = Path(__file__).parent.absolute()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
