import numpy as np
import pandas as pd
import pickle

class Evaluator:

    def __init__(model_path: str, dataset_path: str):
        self.dataset_path = dataset_path
        self.model_path = model_path

        self.dataset = self.loadDataset()
        self.model = self.loadMode()


    def loadModel(self):
        with open(self.model_path, 'rb') as file:
            return pickle.load(file)

    def loadDataset(self):
        return pd.read_csv(self.dataset_path)
    
    def processDataset(self):
        pass

    def computeError(self):
        processed_dataset = self.processDataset()
        
        return 10
