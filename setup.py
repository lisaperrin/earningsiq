from pathlib import Path
from setuptools import setup, find_packages


def read_requirements() -> list[str]:
    """requirements.txt is the single source of truth for dependencies."""
    lines = Path(__file__).with_name("requirements.txt").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


setup(
    name="earningsiq",
    version="0.1.0",
    description="RAG question answering over SEC 10-K/10-Q filings with small local LLMs",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
)
