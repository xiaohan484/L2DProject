import json
from pathlib import Path

jsonFile = Path("shared.json")

def setJsonPath(filePath):
    global jsonFile
    jsonFile = Path(filePath)

def readData():
    global jsonFile
    data = {}
    if jsonFile.is_file():
        with open(jsonFile, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print("no such file, create new")
    return data

def writeData(data):
    global jsonFile
    with open(jsonFile, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def addDataDict(additional:dict):
    global jsonFile
    data = readData()
    for key,content in additional.items():
        data[key] = content
    writeData(data)

def addData(key:str,content):
    global jsonFile
    data = readData()
    data[key] = content
    writeData(data)