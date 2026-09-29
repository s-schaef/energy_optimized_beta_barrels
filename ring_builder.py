#!/usr/bin/env python3
"""Backwards-compatible entry point. Prefer: barrel-build"""
from barrel_builder.ring_builder import *  # noqa: F401,F403
from barrel_builder.ring_builder import main

if __name__ == "__main__":
    main()
