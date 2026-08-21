# conftest.py
# Adds the project root to sys.path so that test files can import app and
# features without needing individual sys.path.append hacks.
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
