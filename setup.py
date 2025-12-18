
from setuptools import setup, find_packages

setup(
    name="openrouter-free",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "openrouter-free = openrouterfree.scanner:main",
            "openrouter-free-proxy = openrouterfree.proxy:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool to fetch and save free models from OpenRouter.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/openrouter-free",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
