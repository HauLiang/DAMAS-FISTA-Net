import yaml


class Config():
    def __init__(self,configFile):
        """ Load the configuration from the specified YAML file """
        self.configFile=configFile

        # Load configuration
        with open(self.configFile,'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def getConfig(self):
        # Retrieve the configuration dictionary
        if self.config:
            return self.config
        else:
            print("input error")
