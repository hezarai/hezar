import setuptools

with open("README.md", "r") as f:
    long_description = f.read()


setuptools.setup(
    name="hezar",
    version="0.1.0",
    url="https://github.com/hezar-ai/hezar",
    license="MIT",
    author="Aryan Shekarlaban",
    author_email="arxyzan@gmail.com",
    description="Hezar: A seamless AI framework & library for Persian",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "omegaconf>=2.3.0",
        "transformers>=4.26",
        "datasets>=2.9.0",
        "huggingface_hub>=0.12.0",
        "pillow",
    ],
    python_requires=">=3.8",
)
