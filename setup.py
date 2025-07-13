from setuptools import setup, find_packages

setup(
    name="mlfs",
    version="0.1.0",
    description="Machine Learning algorithms implemented from scratch",
    author="FW",
    packages=find_packages(),  
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "plotly",
        "matplotlib",
        "memory_profiler"
    ],
    python_requires=">=3.8",
)
