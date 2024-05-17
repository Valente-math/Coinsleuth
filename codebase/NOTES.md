I think my ideal project structure would be:

repo-name
├── .gitignore
├── LICENSE
├── README.md
├── project
│   ├── poetry.lock
│   ├── pyproject.toml
│   ├── README.md
│   ├── src
│   │   └── __init__.py
│   └── tests
│       └── __init__.py


Use the command `poetry new project --name src` to set up a project this way.