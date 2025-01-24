from setuptools import setup, find_packages

setup(
    name="bssnn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.2",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "networkx>=2.5",
        "shap>=0.39.0"
    ],
    author="Emanuele Nardone",
    author_email="emanuele.nardone@unicas.it",
    description="Bayesian State-Space Neural Networks for Interpretable Machine Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Narden91/BSSNN",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)