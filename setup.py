"""Setup script for poet-torch package."""

from setuptools import setup, find_packages
import os

# Read README
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation"

# Read requirements
requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
if os.path.exists(requirements_path):
    with open(requirements_path, "r", encoding="utf-8") as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    install_requires = [
        "torch>=2.7.0",
        "numpy",
        "triton>=3.4.0"
    ]

setup(
    name="poet-torch",
    version="0.0.1",
    description="POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sphere-AI-Lab/poet",
    project_urls={
        "Bug Tracker": "https://github.com/Sphere-AI-Lab/poet/issues",
        "Documentation": "https://github.com/Sphere-AI-Lab/poet/blob/main/README.md",
        "Source": "https://github.com/Sphere-AI-Lab/poet",
    },
    packages=find_packages(include=["poet_torch", "poet_torch.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require={
    },
    keywords=[
        "deep learning",
        "machine learning",
        "neural networks",
        "optimization",
        "llm training",
        "parameter efficient",
        "orthogonal transformation",
        "pytorch",
    ],
    license="BSD 3-Clause License",
    zip_safe=False,
)
