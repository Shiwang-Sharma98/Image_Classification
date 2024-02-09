import os
from pathlib import Path

project_name = "image_classification"

List_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "notebook/train.ipynb",

]


for filepath in List_of_files:
    filepath = Path(filepath)
    filedir,filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)

    if (not os.path.exists(filename)) or (os.path.getsize(filepath)==0):
        with open(filepath,"w") as f:
            pass




