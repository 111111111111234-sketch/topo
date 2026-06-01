from setuptools import setup, find_packages

setup(
    name="conftopo",
    version="0.1.0",
    packages=find_packages(include=["conftopo", "conftopo.*"]),
    install_requires=[
        "numpy>=1.20",
        "networkx>=2.6",
        "matplotlib",
    ],
    extras_require={
        "perception": ["open_clip_torch", "ftfy", "regex"],
        "llm": ["openai"],
        "all": ["open_clip_torch", "ftfy", "regex", "openai", "scipy"],
    },
)
