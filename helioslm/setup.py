from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="helioslm",
    version="1.0.0",
    author="HeliosLM Team",
    author_email="chienhsin@yahoo.com",
    description="HeliosLM v1.0 - P0+P1+P2+P3 Production Stack for Open Frontier Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tonythetiger168/helioslm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "train": ["deepspeed>=0.12.0", "transformers>=4.35.0", "accelerate>=0.24.0"],
        "serve": ["vllm>=0.2.0", "fastapi>=0.104.0", "uvicorn>=0.24.0"],
        "data": ["datasets>=2.14.0", "sentencepiece>=0.1.99", "webdataset>=0.2.0"],
        "rag": ["faiss-cpu>=1.7.4"],
        "dev": ["pytest>=7.4.0", "black>=23.0.0", "flake8>=6.0.0"],
        "all": [
            "deepspeed>=0.12.0", "transformers>=4.35.0", "accelerate>=0.24.0",
            "vllm>=0.2.0", "fastapi>=0.104.0", "uvicorn>=0.24.0",
            "datasets>=2.14.0", "sentencepiece>=0.1.99", "webdataset>=0.2.0",
            "faiss-cpu>=1.7.4", "pytest>=7.4.0", "black>=23.0.0", "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "helioslm-train=scripts.train:main",
            "helioslm-infer=scripts.inference:main",
        ],
    },
)
