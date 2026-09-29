#!/usr/bin/env python3
"""Backwards-compatible entry point. Prefer: barrel-optimize"""
from barrel_builder.optimization import *  # noqa: F401,F403
from barrel_builder.optimization import main

if __name__ == "__main__":
    main()
