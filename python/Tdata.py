"""File to implement Tdata class which is used when
training neural network."""

from lib.utils.exceptions import SizeError
from lib.utils.type import number
import numpy as np
import json

class Tdata:
    """Tdata class is used to store training data which will
    be used to train neural network.
    
    Parameters
    ----------
    `data` : matrix of number, training data instances' feature values.
        default = []
    
    `trgt` : array of number, training data instances' target class. Must
        have the same size as `data`. default = []
    
    Attributes
    ----------
    `size` : int, number of instances.
    
    `data` : matrix of number, training data instances' features value.

    `trgt` : array of number, training data instances' target class.
    """

    def __init__(self, data: [[number]] = [], trgt: [number] = []):
        if len(data) != len(trgt):
            raise SizeError("Number of data doesn't match with number of target")
        
        self.__size = len(data)
        self.__data = data
        self.__trgt = trgt

        # Handling np.ndarray input
        if isinstance(self.__data, np.ndarray):
            self.__data = self.__data.tolist()
        
        if isinstance(self.__trgt, np.ndarray):
            self.__trgt = self.__trgt.tolist()
    

    def get_size(self) -> int:
        """get number of instances from training data.
        
        Parameters
        ----------
        `None`

        Returns
        -------
        `size` : int, number of instances
        """
        return self.__size
    
    def get_instances(self) -> [[number]]:
        """get all instances' features value from
        training data.
        
        Parameters
        ----------
        `None`

        Returns
        -------
        `size` : matrix of number, instances' features value
        """
        return self.__data
    
    def get_instance(self, idx: int) -> [number]:
        """Get an instance from training data with
        given index.
        
        Parameters
        ----------
        `idx` : int, index of instance
        
        Returns
        -------
        `ins` : array of number, instance's features value
        """
        return self.__data[idx]
    
    def get_targets(self) -> [number]:
        """Get all target for every instances from
        training data.
        
        Parameters
        ----------
        `None`
        
        Returns
        -------
        `trgt` : array of number, instances' target class
        """
        return self.__trgt
    
    def get_target(self, idx: int) -> number:
        """Get a target for an instance with given
        index from training data.
        
        Parameters
        ----------
        `idx` : int, index of instance

        Returns
        -------
        `trgt` : number, instance's target class
        """
        return self.__trgt[idx]


    def save(self, fileName: str) -> None:
        """Save training data (data & target) to an external .json file.
        
        Parameters
        ----------
        `fileName` : str, target filename. must include '.json' extension
        
        Returns
        -------
        `None`
        """
        data = {
            "size" : self.__size,
            "data" : self.__data,
            "trgt" : self.__trgt
        }
        
        jsonObj = json.dumps(data, indent=2)
        with open(fileName, 'w') as file:
            file.write(jsonObj)
    
    def load(self, fileName: str) -> None:
        """Load training data (data & target) from an external .json file.
        
        Parameters
        ----------
        `fileName` : str, target filename. must include '.json' extension

        Returns
        -------
        `None`
        """
        with open(fileName, 'r') as file:
            jsonObj = json.load(file)
            
            self.__size = jsonObj['size']
            self.__data = jsonObj['data']
            self.__trgt = jsonObj['trgt']