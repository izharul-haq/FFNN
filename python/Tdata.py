from python.utils.exceptions import SizeError
import numpy as np
import json

class Tdata:
    """Tdata class is used to store training data which will
    be used to train neural network.
    
    Parameters
    ----------
    `data` : ndarray, training data instances' feature values.
        default = []
    
    `trgt` : ndarray, training data instances' target class. Must
        have the same size as `data`. default = []
    
    Attributes
    ----------
    `size` : int, number of instances.
    
    `data` : ndarray, training data instances' features value.

    `trgt` : ndarray, training data instances' target class.

    Raises
    ------
    `SizeError`
        raised when number of data doesn't match with number of target.
    """
    
    def __init__(self, data: np.ndarray = np.array([]), trgt: np.ndarray = np.array([])):
        if len(data) != len(trgt):
            raise SizeError("Number of item doesn't match")

        self.__size = len(data)
        self.__data = data
        self.__trgt = trgt
        
        if trgt.size != 0:
            self.__trgt = self.__trgt.reshape((self.__size, trgt.size // self.__size))


    def get_size(self) -> int:
        """Get number of instances from training data.
        
        Returns
        -------
        int,
            number of instances
        """
        return self.__size
    
    def get_instances(self) -> np.ndarray:
        """get all instances' features value from
        training data.
        
        Returns
        -------
        ndarray,
            instances' features value
        """
        return self.__data
    
    def get_instance(self, idx: int) -> np.ndarray:
        """Get an instance from training data with
        given index.
        
        Parameters
        ----------
        `idx` : int,
            index of instance
        
        Returns
        -------
        ndarray,
            instance's features value
        """
        return self.__data[idx]
    
    def get_targets(self) -> np.ndarray:
        """Get all target for every instances from
        training data.
        
        Returns
        -------
        ndarray,
            instances' target class
        """
        return self.__trgt
    
    def get_target(self, idx: int) -> np.ndarray:
        """Get a target for an instance with given
        index from training data.
        
        Parameters
        ----------
        `idx` : int,
            index of instance

        Returns
        -------
        ndarray,
            instance's target class
        """
        return self.__trgt[idx]

    def get_some(self, start_idx: int, end_idx: int) -> 'Tdata':
        """ Get a portion of dataset starting from `start_idx`
        to `end_idx` (half-open).
        
        Parameters
        ----------
        `start_idx` : int,
            first index
        
        `end_idx` : int,
            last index

        Returns
        -------
            Tdata,
                a portion of dataset
        """
        new_data = self.__data[start_idx: end_idx]
        new_trgt = self.__trgt[start_idx: end_idx]

        return Tdata(new_data, new_trgt)


    def save(self, file_name: str, path: str = "") -> None:
        """Save training data (data & target) to an external `.json` file.
        
        Parameters
        ----------
        `file_name` : str,
            name of file to write. If such file doesn't exist a
            new file will be created. Must include `.json`
            extension

        `path` : str, optional,
            existing path to write file, by default "". Example:
            `../json`
        """
        data = {
            "size" : self.__size,
            "data" : self.__data.tolist(),
            "trgt" : self.__trgt.tolist()
        }
        
        json_obj = json.dumps(data, indent=4)
        with open(path + '/' + file_name, 'w') as file:
            file.write(json_obj)
    
    def load(self, file_path: str) -> None:
        """Load training data (data & target) from an external `.json` file.
        
        Parameters
        ----------
        `file_path` : str,
            path of file to read. If such file doesn't exist a
            FileNotFoundError will raise. Must include `.json`
            extension
        """
        with open(file_path, 'r') as file:
            json_obj = json.load(file)
            
            self.__size = json_obj['size']
            self.__data = np.array(json_obj['data'])
            self.__trgt = np.array(json_obj['trgt'])