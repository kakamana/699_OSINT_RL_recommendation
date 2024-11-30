from setuptools import setup, find_packages

setup(
    name="osint_recommendation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pandas",
        "numpy",
        "scikit-learn",
        "fastapi",
        "uvicorn"
    ],
)