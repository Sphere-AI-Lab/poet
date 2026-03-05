from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="poet-torch",
    version="1.0",
    description="POET: Memory-Efficient LLM Training via Parameterized Orthogonal Expansion",
    url="https://github.com/Sphere-AI-Lab/poet.git",
    author="Zeju Qiu",
    author_email="zeju.qiu@gmail.com",
    license="Apache 2.0",
    packages=["poet_torch"],
    install_requires=required,
)