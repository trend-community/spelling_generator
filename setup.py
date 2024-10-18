from setuptools import setup, find_packages

setup(
    name="ipa_inverter",
    version="0.1.0",
    description="A package for spelling variation generation using LangChain",
    author="Colton Flowers",
    author_email="cflowers@trend.community",
    packages=find_packages(),
    install_requires=[
        "langchain-core",
        "langchain-openai",
        "langchain-cache",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)