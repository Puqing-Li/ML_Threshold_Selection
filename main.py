#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application entrypoint - delegates to App Controller.
"""

import tkinter as tk
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from ml_threshold_selection.app_controller import FixedMLGUI


def main():
    root = tk.Tk()
    app = FixedMLGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
