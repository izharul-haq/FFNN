"""File to implement Neural class which is
implemented neural network."""

from python.utils.exceptions import SizeError
from python.Tdata import Tdata
from graphviz import Graph
from math import exp
from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

class Neural:
    """Neural class is used to define feedforward neural network (FFNN)
    model.
    
    Parameters
    ----------
    `neuron_each_layer` : List[int], number of neurons for each layer.
        default value = []. First item of this parameter refer to number
        of neurons in input layer, last item refer to number of neurons in
        output layer, and the rest refer to number of neurons in hidden
        layer(s).
    
    `random_weight` : bool, decide wether weights are initialized randomly
        or not. default value = True. If to set False, each weight is
        initialized with default value 0.
    
    `weight_range` : Tuple[float, float], range of initialized weight value
        (upper bound exclusive). default value = (-1, 1). Only used if
        `random_weight` is set to True.

    `random_bias` : bool, decide wether biases are initialized randomly
        or not. default value = False. If set to False, each bias is
        initialized with default value 1. Else, each bias is initialized
        with random value between -1 and 1 (upper bound exclusive).
    
    Attributes
    ----------
    `depth` : int, number of layers from neural network.

    `neuron_each_layer` : List[int], number of neurons for each layer.

    `weights` : List[List[float]], neural network weights for each connection
        between i-th layer to (i+1)-th layer.

    `biases` : List[float], bias for each layer (exc. output layer).

    `activation_funcs` : List[str], list of activation function acronym for
        each layer (exc. input layer). Default activation for each layer is
        sigmoid for each layer. Acronym for each activation function:  
    
        - 'none' : none
        - 'sigm' : sigmoid
        - 'relu' : ReLu
        - 'linr' : linear
        - 'sfmx' : softmax
    """

    def __init__(self, neuron_each_layer: List[int] = [], random_weight: bool = True, weight_range: Tuple[float, float] = (-1, 1), random_bias: bool = False):
        self.__depth = len(neuron_each_layer)
        self.__neuron_each_layer = neuron_each_layer
        self.__weights = []
        self.__biases = []
        self.__activation_funcs = ["none"] + ["sigm" for i in range(self.__depth - 1)]
        
        # Map to get each activation function easier
        self.__activationMap = {"sigm": self.__sigmoid, "linr": self.__linear,
                                "relu": self.__reLU, "sfmx": self.__softmax}
        
        for i in range(self.__depth - 1):
            # Initializing i-th layer
            self.__weights.append([])
            
            for j in range(self.__neuron_each_layer[i] + 1):
                # Initializing j-th neuron in i-th layer, 0-th neuron is bias
                self.__weights[i].append([])

                for k in range(self.__neuron_each_layer[i + 1]):
                    # initializing weight
                    w = 0
                    if random_weight:
                        w = np.random.uniform(weight_range[0], high=weight_range[1])
                    
                    self.__weights[i][j].append(w)
        
        for i in range(self.__depth - 1):
            # Initializing i-th bias
            b = 1
            if random_bias:
                b = np.random.uniform(-1, high=1)
            
            self.__biases.append(b)
    
    
    def get_depth(self) -> int:
        """Get number of layers from network model.

        Returns
        -------
        int,
            number of layers
        """
        return self.__depth
    
    def get_weights_from_layer(self, i: int) -> List[List[float]]:
        """Get list of weights from connection between
        i-th layer and (i+1)-th layer.
        
        Parameters
        ----------
        `i` : int,
            index of layer
        
        Returns
        -------
        List[List[float]],
            list of weight between i-th and (i+1)-th layer
        """
        return self.__weights[i]
    
    def get_num_of_features(self) -> int:
        """Get number of features (input) from network model.
        
        Returns
        -------
        int,
            number of neurons in input layer        
        """
        return self.__neuron_each_layer[0]
    
    def set_weight(self, idx: int, neuron_src: int, neuron_dst: int, value: float) -> None:
        """Set weight of connection between neuron_src in
        given layer to neuron_dst in next layer.
        
        Parameters
        ----------
        `idx` : int,
            index of layer
            
        `neuron_src` : int,
            index of neuron in `idx` layer
            
        `neuron_dst` : int,
            index of neuron in next layer
            
        `value` : float,
            weight value
        """
        self.__weights[idx][neuron_src][neuron_dst] = value
    
    def set_activation_funcs(self, func: List[str]) -> None:
        """Set activation function for all layers.
        
        Parameters
        ----------
        `func` : List[str],
            list of acronym of activation function for
            each layer
        """
        self.__activation_funcs = func

    def set_layer_activation_funcs(self, layer: int, func: str) -> None:
        """Set activation function for given layer.
        
        Parameters
        ----------
        `layer` : int,
            index of layer
            
        `func` : str,
            acronym of activation value
        """
        self.__activation_funcs[layer] = func


    def __sum(self, inputs: List[List[float]], weights: List[List[float]]) -> List[List[float]]:
        """Return summation function for each node with given inputs
        and weights.
        
        Parameters
        ----------
        `inputs` : List[List[float]],
            input matrix
            
        `weights` : List[List[float]],
            weight matrix
            
        
        Returns
        -------
        List[List[float]],
            result from `inputs` x `weights`
        
        Raises
        ------
        `SizeError`
            raised when `inputs` column doesn't match
            with `weights` row.
        """
        if (len(inputs[0]) != len(weights)):
            raise SizeError("dimention from each matrix doesn't match")

        mat01 = np.array(inputs)
        mat02 = np.array(weights)
        
        res = np.dot(mat01, mat02)
        
        return list(res)

    def __linear(self, x: float) -> float:
        """Calculate linear activation value from given x
        with f(x) = 1 * x.
        
        Parameters
        ----------
        `x` : float,
            input for linear activation function
        
        Returns
        -------
        float,
            result of linear activation function
        """
        return x

    def __linear_prime(self, x: float) -> float:
        """Calculate linear derivative value from given x
        with f(x) = 1.
        
        Parameters
        ----------
        `x` : float,
            input for linear derivative function
        
        Returns
        -------
        float,
            result of linear derivative function
        """        
        return 1
    
    def __sigmoid(self, x: float) -> float:
        """Calculate sigmoid activation value from given x
        with f(x) = 1 / (1 + exp(-x)).

        Parameters
        ----------
        `x` : float,
            input for sigmoid activation function
        
        Returns
        -------
        float,
            result of sigmoid activation function
        """
        return 1 / (1 + exp(-x))
    
    # TODO : implement sigmoid derivative function
    def __sigmoid_prime(self, x: float) -> float:
        """Calculate sigmoid derivative value from given x
        with f'(x) = f(x) * (1 - f(x)).
        
        Parameters
        ----------
        `x` : float,
            input for sigmoid derivative function
        
        Returns
        -------
        float,
            result of sigmoid derivative function
        """
        pass
    
    def __reLU(self, x: float) -> float:
        """Calculate ReLU activation value from given x
        with f(x) = max(0, x).
        
        Parameters
        ----------
        `x` : float,
            input for ReLU activation function
        
        Returns
        -------
        float,
            result of ReLU activation function
        """
        return max(0, x)
    
    # TODO : implement sigmoid derivative function
    def __reLU_prime(self, x: float) -> float:
        """Calculate ReLU derivative value from given x
        with f'(x) = 1 if x > 0 else 0.
        
        Parameters
        ----------
        `x` : float,
            input for ReLU derivative function    
        
        Returns
        -------
        float,
            result of ReLU derivative function
        """        
        pass

    def __softmax(self, x: List[float]) -> List[float]:
        """Calculate softmax activation value from given x
        with f(xi) = exp(xi) / sum(exp(xi)); i = 0, 1, ..., K.
        
        Parameters
        ----------
        `x` : List[float],
            inputs for softmax activation function
        
        Returns
        -------
        List[float],
            result of softmax activation function
        """
        res = []
        denominator = 0
        for i in range(len(x)):
            denominator += exp(x[i])
        
        for i in range(len(x)):
            res.append(exp(x[i]) / denominator)

        return res
    
    def __sigma(self, layer: int, x: Union[float, List[float]]) -> List[float]:
        """Calculate activation function value for each neuron from
        given layer.
        
        Parameters
        ----------
        `layer` : int,
            index of layer
            
        `x` : Union[float, List[float]],
            input(s) for activation function. If activation value is
            softmax, type for `x` is List[float], else float.
        
        Returns
        -------
        List[float],
            list of activation function value for each neuron from
            given layer 
        """
        res = []
        actFunc = self.__activation_funcs[layer]
        if actFunc == "sfmx":
            res = self.__softmax(x)
        else:
            func = self.__activationMap[actFunc]
            for i in range(len(x)):
                res.append(func(x[i]))
        
        return res

    
    def forward(self, instance: List[float]) -> List[float]:        
        """Forward propagation implementation with given
        instance's data.
        
        Parameters
        ----------
        `instance` : List[float],
            instance's features data
        
        Returns
        -------
        List[float],
            list of results from output layer.
        """
        inp = instance
        for j in range(1, self.__depth):
            # Create inputs matrix (bias value + input for each layer)
            # and weights matrix
            inp = [[self.__biases[j-1]] + inp]
            weights = self.get_weights_from_layer(j-1)
            
            # Matrix multiplication (inputs X weights)
            summ = self.__sum(inp, weights)

            # Update results with activation function for this layer
            summ[0] = self.__sigma(j, summ[0])

            # Set activation result as input for another layer
            inp = summ[0]

        return inp

    # TODO : implement backward propagation section
    def fit(self, training_data: Tdata, batch_size: int = 1, learning_rate: float = 0.1, threshold: float = 0.001) -> None:
        """Backpropagation algorithm implementation. Used by neural
        network to learn from given training data.
        
        Parameters
        ----------
        `trainingData` : Tdata,
            Training data used to learn.
            
        `batchSize` : int, optional,
            [description], by default 1
            
        `learningRate` : float, optional,
            [description], by default 0.1
            
        `threshold` : float, optional,
            [description], by default 0.001
        """ 
        pass
    
    def save(self, file_name: str,  path: str = "") -> None:
        """Save neural network model to an external `.json` file.
        
        Parameters
        ----------
        `file_name` : str,
            name of file to write. If such file doesn't exist a new
            file will be created. Must include `.json` extension
            
        `path` : str, optional,
            existing path to write file, by default "". Example:
            `..\json`
        """
        neural = {
            "depth" : self.__depth,
            "neuron_each_layer" : self.__neuron_each_layer,
            "weights" : self.__weights,
            "biases" : self.__biases,
            "activation_funcs" : self.__activation_funcs
        }
        
        json_obj = json.dumps(neural, indent=4)
        with open(path + '\\' + file_name, 'w') as file:
            file.write(json_obj)
    
    def load(self, file_path: str) -> None:
        """Load neural network model from an external `.json` file.
        
        Parameters
        ----------
        `file_path` : str,
            path of file to read. If such path doesn't exist a
            FileNotFoundError will raise. Must include `.json`
            extension. Example : `../json/neural.json`
        """
        with open(file_path, 'r') as file:
            json_obj = json.load(file)
            
            self.__depth = json_obj['depth']
            self.__neuron_each_layer = json_obj['neuron_each_layer']
            self.__weights = json_obj['weights']
            self.__biases = json_obj['biases']
            self.__activation_funcs = json_obj['activation_funcs']

    def show(self, file_name: str, view: bool = False) -> None:
        """Show neural network model in graph form. This function
        will create `.png` file in `img` folder" relative to file.
        
        Parameters
        ----------
        `file_name` : str,
            name of `.png` file
            
        `view` : bool, optional,
            tell wether model will be immediately opened by default
            image viewer or not, by default False
        """
        # Initialize graph
        nn_model = Graph()
        nn_model.attr(rankdir='LR')
        nn_model.attr(splines='line')
        
        # Initialize nodes
        for i in range(self.__depth):
            # Initializing subgraph for each layer
            sub_graph = Graph(name='cluster_' + str(i))
            sub_graph.attr(color='none')
            
            nodeLabel = ''
            if i == 0:
                nodeLabel = 'x'
            elif i == self.__depth - 1:
                nodeLabel = 'o'
            else:
                nodeLabel = 'h' + str(i)
            
            # Adding nodes to subgraph
            for j in range(self.__neuron_each_layer[i] + 1):
                node = nodeLabel + str(j)
                if j == 0:
                    if nodeLabel == 'o':
                        pass
                    else:
                        sub_graph.node(node, node, shape='rect')
                else:
                    sub_graph.node(node, node, shape='circle')
            
            # Append subgraph to main graph
            nn_model.subgraph(sub_graph)
            
        # Initialize edges
        for i in range(self.__depth - 1):
            nodeFrom = ''
            nodeTrgt = ''
            if i == 0:
                nodeFrom = 'x'
                nodeTrgt = 'h' + str(i+1)
            elif i == self.__depth - 2:
                nodeFrom = 'h' + str(i)
                nodeTrgt = 'o'
            else:
                nodeFrom = 'h' + str(i)
                nodeTrgt = 'h' + str(i+1)
            
            # Initialize edge between j-th neuron from i-th layer to k-th neuron in (i+1)-th layer 
            for j in range(self.__neuron_each_layer[i] + 1):
                for k in range(1, self.__neuron_each_layer[i+1] + 1):
                    weight = self.__weights[i][j][k - 1]
                    nn_model.edge(nodeFrom + str(j), nodeTrgt + str(k), xlabel=str(weight), minlen='5')
        
        # render image
        imgPath = nn_model.render(filename=file_name, directory='img/', view=view, format='png', cleanup=True)