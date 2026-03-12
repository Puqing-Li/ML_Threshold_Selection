#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application entrypoint - delegates to App Controller.
"""

import tkinter as tk
from src.ml_threshold_selection.app_controller import FixedMLGUI


def main():
    root = tk.Tk()
    app = FixedMLGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
