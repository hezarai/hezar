from setuptools import setup

setup(
    name='hezar',
    version='0.1.0',
    install_requires=[
        "torch>=1.10.0"
        "omegaconf>=2.3.0"
        "transformers>=4.26",
        "huggingface_hub>=0.12.0"
    ],
    url='https://github.com/hezar-ai/hezar',
    license='MIT',
    author='Aryan Shekarlaban',
    author_email='arxyzan@gmail.com',
    description='Hezar: A seamless AI framework & library for Persian'
)
