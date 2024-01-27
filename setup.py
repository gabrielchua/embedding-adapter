from setuptools import setup, find_packages

setup(
    name='embedding_adapter',
    version='0.1.0',
    author='Gabriel Chua',
    author_email='cyzgab@gmail.com',
    description='A lightweight open-source package to fine-tune embedding models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'tqdm',
        'pydantic',
        'openai'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
