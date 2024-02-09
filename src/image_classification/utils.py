import os
from box.exceptions import BoxValueError
import yaml
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
        return ConfigBox(content)
    
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
  


@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)






@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)

   
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
 
    joblib.dump(value=data, filename=path)
 


@ensure_annotations
def load_bin(path: Path) -> Any:
 
    data = joblib.load(path)
 
    return data

@ensure_annotations
def get_size(path: Path) -> str:

    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())