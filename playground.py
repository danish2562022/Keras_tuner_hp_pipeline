import yaml
with open('config.yaml') as f:
    config = yaml.load(f, Loader = yaml.FullLoader) 
x =".".join(config['input_files']['models'].split('/')[0:-1])+"."+config['input_files']['models'].split("/")[-1].split(".")[0] + " import *"
print(x.split(" ")[-3].split(".")[-1])