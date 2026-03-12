#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight logging adapter for GUI logger.
"""

from typing import Callable


class UILogger:
    def __init__(self, sink: Callable[[str], None]):
        self._sink = sink

    def info(self, msg: str):
        self._sink(msg)

    def step(self, msg: str):
        self._sink(msg)

    def error(self, msg: str):
        self._sink(msg)
