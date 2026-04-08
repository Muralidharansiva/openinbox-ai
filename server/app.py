[project]
name = "openinbox-ai"
version = "0.1.0"
description = "AI email triage environment"
authors = [{name = "MDSon07"}]

dependencies = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "openenv-core>=0.2.0"
]

[project.scripts]
server = "server.app:app"

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"