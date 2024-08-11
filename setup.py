from setuptools import setup, find_packages

setup(
    name='emotion_recognition',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'torchaudio',
        'transformers',
        'librosa',
        'azure-storage-blob',
    ],
)