#!/usr/bin/env python3
import sys

if sys.version_info < (3, 9):
    sys.exit('ERROR: Backtesting.py requires Python 3.9+')

if __name__ == '__main__':
    from setuptools import setup
    setup()

