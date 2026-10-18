from setuptools import setup, find_packages

setup(
    name="agentaudit",
    version="1.0.0",
    description="Deterministic Stress-Testing & Trajectory DPO Harness for Multi-Turn Frontier Agents",
    packages=find_packages(),
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "agentaudit=agentaudit.cli:main",
        ],
    },
)
