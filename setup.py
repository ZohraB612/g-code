#!/usr/bin/env python3
"""
Setup script for gcode - Your intelligent coding companion.
Install globally with: pip install -e .
Then use anywhere with: gcode
"""

from setuptools import setup, find_packages

# Read README.md from current directory
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements.txt from current directory
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gcode",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Your intelligent coding companion - like Claude Code but with dual API support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gcode",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Testing",
        "Topic :: Software Development :: Quality Assurance",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gcode=gcode.agent:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
