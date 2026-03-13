from setuptools import setup, find_packages

setup(
    name="flocking",
    version="0.1.0",
    description="Emergent flocking behavior via Deep Double DQN",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "agentpy",
        "gymnasium",
        "pfrl",
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "PyYAML>=6.0",
    ],
)
