from pathlib import Path
import yaml

#https://stackoverflow.com/questions/6866600/how-to-parse-read-a-yaml-file-into-a-python-object
class Struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

def getModelConfig():
    with open(Path(__file__).parent / 'config.yml', 'r') as yamlfile:
        configContents = yaml.safe_load(yamlfile)
    return Struct(**configContents)