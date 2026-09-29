#!/usr/bin/env python3
"""Backwards-compatible entry point. Prefer: barrel-align"""
from barrel_builder.alignment import *  # noqa: F401,F403
from barrel_builder.alignment import main

if __name__ == "__main__":
    main()
