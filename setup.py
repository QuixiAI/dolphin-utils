from setuptools import setup, find_packages

setup(
    name="dolphin-summarize",
    version="0.3.0",
    description="Summarize model architecture from safetensors files",
    author="Eric",
    packages=find_packages(),
    install_requires=[
        # No strict requirements
    ],
    extras_require={
        "full": ["safetensors", "huggingface_hub"],  # Optional dependencies
        "hub": ["huggingface_hub"],  # For Hugging Face Hub support only
    },
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "dolphin-summarize=dolphin_summarize.cli:main",
        ],
    },
)
