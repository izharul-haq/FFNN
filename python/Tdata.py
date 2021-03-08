"""File to implement Tdata class which is used when
training neural network."""

from python.utils.exceptions import SizeError
from typing import List
import numpy as np
import json

class Tdata:
    """Tdata class is used to store training data which will
    be used to train neural network.
    
    Parameters
    ----------
    `data` : List[List[float]], training data instances' feature values.
        default = []
    
    `trgt` : List[float], training data instances' target class. Must
        have the same size as `data`. default = []
    
    Attributes
    ----------
    `size` : int, number of instances.
    
    `data` : List[List[float]], training data instances' features value.

    `trgt` : List[float], training data instances' target class.

    Raises
    ------
    `SizeError`
        raised when number of data doesn't match with number of target.
    """

    def __init__(self, data: List[List[float]] = [], trgt: List[float] = []):
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
        """Get number of instances from training data.
        
        Returns
        -------
        int,
            number of instances
        """
        return self.__size
    
    def get_instances(self) -> List[List[float]]:
        """get all instances' features value from
        training data.
        
        Returns
        -------
        List[List[float]],
            instances' features value
        """
        return self.__data
    
    def get_instance(self, idx: int) -> List[float]:
        """Get an instance from training data with
        given index.
        
        Parameters
        ----------
        `idx` : int,
            index of instance
        
        Returns
        -------
        List[float],
            instance's features value
        """
        return self.__data[idx]
    
    def get_targets(self) -> List[float]:
        """Get all target for every instances from
        training data.
        
        Returns
        -------
        List[float],
            instances' target class
        """
        return self.__trgt
    
    def get_target(self, idx: int) -> float:
        """Get a target for an instance with given
        index from training data.
        
        Parameters
        ----------
        `idx` : int,
            index of instance

        Returns
        -------
        float,
            instance's target class
        """
        return self.__trgt[idx]


    def save(self, file_name: str, path: str = "") -> None:
        """Save training data (data & target) to an external `.json` file.
        
        Parameters
        ----------
        `file_name` : str,
            name of file to write. If such file doesn't exist a new
            file will be created. Must include `.json` extension
            
        `path` : str, optional,
            existing path to write file, by default ""
        """
        data = {
            "size" : self.__size,
            "data" : self.__data,
            "trgt" : self.__trgt
        }
        
        json_obj = json.dumps(data, indent=2)
        with open(path + file_name, 'w') as file:
            file.write(json_obj)
    
    def load(self, file_path: str) -> None:
        """Load training data (data & target) from an external `.json` file.
        
        Parameters
        ----------
        `file_path` : str,
            path of file to read. If such path doesn't exist a
            FileNotFoundError will raise. Must include `.json`
            extension. Example : `../json/neural.json`
        """
        with open(file_path, 'r') as file:
            json_obj = json.load(file)
            
            self.__size = json_obj['size']
            self.__data = json_obj['data']
            self.__trgt = json_obj['trgt']