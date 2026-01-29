from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mechanex",
    version="0.5.0",
    author="Axionic Labs",
    author_email="contact@axioniclabs.ai",
    description="A Python client for the Axionic API.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://axioniclabs.ai",
    packages=find_packages(),
    install_requires=[
        "requests>=2.20.0",
        "ipython",
        "jupyter",
        "matplotlib",
        "click",
        "fastapi",
        "uvicorn",
        "torch",
        "numpy",
        "tqdm",
        "mdmm",
        "rich",
        "sae-lens",
        "openai",
        "huggingface_hub",
    ],
    entry_points={
        "console_scripts": [
            "mechanex=mechanex.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
