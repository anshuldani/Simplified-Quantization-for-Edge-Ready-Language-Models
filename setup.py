from setuptools import setup, find_packages

setup(
    name="salient-quant",
    version="0.1.0",
    description="Simplified Quantization for Edge-Ready Language Models",
    author="Anshul Dani, Rohit Lahori",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "datasets>=2.18.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
)
