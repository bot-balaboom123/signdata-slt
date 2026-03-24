#!/usr/bin/env python3
"""Convenience wrapper for running the signdata pipeline."""

import sys
import os

# Add src/ to path so signdata is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from signdata.__main__ import main

if __name__ == "__main__":
    main()
