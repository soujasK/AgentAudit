from setuptools import setup, find_packages

setup(
    name="agentfail-bench",
    version="1.0.0",
    description="Deterministic Stress-Testing & Trajectory DPO Harness for Multi-Turn Frontier Agents",
    packages=find_packages(),
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "agentfail=agentfail.cli:main",
        ],
    },
)
