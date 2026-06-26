"""
tests/conftest.py — Cấu hình pytest toàn cục
"""

import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import si_converter
sys.path.insert(0, str(Path(__file__).parent.parent))
