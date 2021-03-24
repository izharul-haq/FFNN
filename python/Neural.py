from python.utils.exceptions import SizeError
from python.Tdata import Tdata
from graphviz import Graph
from math import exp, inf
from typing import List, Tuple
import numpy as np
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

    `weights` : List[ndarray], neural network weights for each connection
        between i-th layer to (i+1)-th layer.

    `biases` : ndarray, bias for each layer (exc. output layer).

    `activation_funcs` : List[str], list of activation function acronym for
        each layer (exc. input layer). Default activation for each layer is
        sigmoid for each layer. Acronym for each activation function:  
    
        - 'none' : none
        - 'sigm' : sigmoid
        - 'relu' : ReLU
        - 'lelu' : leaky ReLU
        - 'linr' : linear
        - 'sfmx' : softmax
    """

    def __init__(self, neuron_each_layer: List[int] = [0, 0], random_weight: bool = True, weight_range: Tuple[float, float] = (-1, 1), random_bias: bool = False):
        self.__depth = len(neuron_each_layer)
        self.__neuron_each_layer = neuron_each_layer
        self.__weights = []
        self.__biases = np.zeros(self.__depth - 1)
        self.__activation_funcs = ["none"] + ["sigm" for i in range(self.__depth - 1)]
        
        # Map to get each activation function easier
        self.__activation_map = {"sigm": self.__sigmoid, "linr": self.__linear,
                                "relu": self.__reLU, "sfmx": self.__softmax}
        
        for i in range(self.__depth - 1):
            # Initializing i-th layer
            layer_src = self.__neuron_each_layer[i] + 1
            layer_dst = self.__neuron_each_layer[i+1]
            self.__weights.append(np.zeros([layer_src, layer_dst]))
            
            for j in range(self.__neuron_each_layer[i]+1):
                for k in range(self.__neuron_each_layer[i + 1]):
                    w = 0
                    if random_weight:
                        w = np.random.uniform(weight_range[0], high=weight_range[1])
                    
                    self.__weights[i][j][k] = w
        
        for i in range(self.__depth - 1):
            # Initializing i-th bias
            b = 1
            if random_bias:
                b = np.random.uniform(-1, high=1)
            
            self.__biases[i] = b
    
    
    def get_depth(self) -> int:
        """Get number of layers from network model.

        Returns
        -------
        int,
            number of layers
        """
        return self.__depth
    
    def get_weights_from_layer(self, i: int) -> np.ndarray:
        """Get list of weights from connection between
        i-th layer and (i+1)-th layer.
        
        Parameters
        ----------
        `i` : int,
            index of layer
        
        Returns
        -------
        ndarray,
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


    def __linear(self, inputs: np.ndarray) -> np.ndarray:
        """Calculate linear activation value for each element
        in inputs with f(x) = 1 * x.
        
        Parameters
        ----------
        `inputs` : ndarray,
            input for linear activation function
        
        Returns
        -------
        ndarray,
            result of linear activation function
        """
        return inputs

    def __linear_prime(self, inputs: np.ndarray) -> np.ndarray:
        """Calculate linear derivative value for each element
        in inputs with f(x) = 1.
        
        Parameters
        ----------
        `inputs` : ndarray,
            input for linear derivative function
        
        Returns
        -------
        ndarray,
            result of linear derivative function
        """        
        return np.where(inputs == inputs, 1, 0)
    
    def __sigmoid(self, inputs: np.ndarray) -> np.ndarray:
        """Calculate sigmoid activation value for each element
        in inputs with f(x) = 1 / (1 + exp(-x)).

        Parameters
        ----------
        `inputs` : ndarray,
            input for sigmoid activation function
        
        Returns
        -------
        ndarray,
            result of sigmoid activation function
        """
        return 1 / (1 + np.exp(-inputs))
    
    def __sigmoid_prime(self, inputs: np.ndarray) -> np.ndarray:
        """Calculate sigmoid derivative value for each element
        in inputs with f'(x) = f(x) * (1 - f(x)).
        
        Parameters
        ----------
        `inputs` : ndarray,
            input for sigmoid derivative function
        
        Returns
        -------
        ndarray,
            result of sigmoid derivative function
        """
        temp = self.__sigmoid(inputs)
        return  temp * (1 - temp)
    
    def __reLU(self, inputs: np.ndarray) -> np.ndarray:
        """Calculate ReLU activation value for each element
        in inputs with f(x) = max(0, x).
        
        Parameters
        ----------
        `inputs` : ndarray,
            input for ReLU activation function
        
        Returns
        -------
        ndarray,
            result of ReLU activation function
        """
        return np.maximum(0, inputs)
    
    def __reLU_prime(self, inputs: np.ndarray) -> np.ndarray:
        """Calculate ReLU derivative value for each element
        in inputs with f'(x) = 1 if x > 0 else 0.
        
        Parameters
        ----------
        `inputs` : ndarray,
            input for ReLU derivative function    
        
        Returns
        -------
        ndarray,
            result of ReLU derivative function
        """        
        return np.where(inputs > 0, 1, 0)
    
    def __leLU(self, inputs: np.ndarray, a: float = 0.001) -> np.ndarray:
        """Calculate leaky ReLU activation value for each element
        in inputs with f(x) = max(a * x, x).
        
        Parameters
        ----------
        `inputs` : ndarray,
            input for leaky ReLU activation function
            
        `a` : float, optional,
            leak constant, by default 0.001
        
        Returns
        -------
        ndarray,
            result of leaky ReLU activation function
        """
        return np.maximum(a * inputs, inputs)
    
    def __leLU_prime(self, inputs: np.ndarray, a: float = 0.001) -> np.ndarray:
        """Calculate leaky ReLU derivative value for each element
        in inputs with f'(x) = 1 if x > 0 else a.
        
        Parameters
        ----------
        `inputs` : ndarray,
            input for leaky ReLU derivative function
            
        `a` : float, optional,
            leak constant, by default 0.001
        
        Returns
        -------
        ndarray,
            result of leaky ReLU derivative function
        """
        return np.where(inputs > 0, 1, a)

    def __softmax(self, inputs: np.ndarray) -> np.ndarray:
        """Calculate softmax activation value for each element
        in inputs with f(xi) = exp(xi) / sum(exp(xi)); i = 0, 1, ..., K.
        
        Parameters
        ----------
        `inputs` : ndarray,
            inputs for softmax activation function
        
        Returns
        -------
        ndarray,
            result of softmax activation function
        """
        res = np.exp(inputs)
        return res / res.sum()

    def __do_per_dnet(self, inputs: np.ndarray, act: str) -> np.ndarray:
        """Calculate do/dnet for each activation function with
        given input.
        
        Parameters
        ----------
        `inputs` : ndarray,
            input for derivative activation function
            
        `act` : str,
            Activaion function for this layer
        
        Returns
        -------
        ndarray,
            Result of derivative activation function
        """
        if act == "sigm":
            return self.__sigmoid_prime(inputs)
        elif act == "relu":
            return self.__reLU_prime(inputs)
        elif act == "lelu":
            return self.__leLU_prime(inputs)
        elif act == "linr":
            return self.__linear_prime(inputs)

    
    def forward(self, instance: np.ndarray) -> np.ndarray:        
        """Forward propagation implementation with given
        instance's data.
        
        Parameters
        ----------
        `instance` : ndarray,
            instance's features data
        
        Returns
        -------
        ndarray,
            list of results from output layer.
        """
        self.__neuron_output = [instance]
        self.__sum = [None]
        inp = instance
        for j in range(1, self.__depth):
            # Create inputs matrix (bias value + input for each layer)
            # and weights matrix
            bias = np.array([self.__biases[j-1]])
            inp = np.concatenate((bias, inp), axis=None)
            weights = self.get_weights_from_layer(j-1)
            
            # Matrix multiplication (inputs X weights)
            summ = np.dot(inp, weights)
            #print(summ)

            # Save summation result (neuron net) for backward propagation
            self.__sum.append(summ)

            # Update results with activation function for this layer
            act = self.__activation_map[self.__activation_funcs[j]]
            summ = act(summ)

            # Save activation result (neuron output) for backward propagation
            self.__neuron_output.append(summ)

            # Set activation result as input for another layer
            inp = summ

        return inp

    # TODO : fixing backward propagation
    def fit(self, training_data: Tdata, batch_size: int = 1, learning_rate: float = 0.1, threshold: float = 0.001, max_iter: int = 100) -> None:
        """Backpropagation algorithm implementation. Used by neural
        network to learn from given training data.
        
        Parameters
        ----------
        `training_data` : Tdata,
            training data instance

        `batch_size` : int, optional,
            number of step before updating weight, by default 1
            
        `learning_rate` : float, optional,
            neural network model learning rate, by default 0.1
            
        `threshold` : float, optional,
            minimum error value, by default 0.001
            
        `max_iter` : int, optional,
            maximum number of iteration, by default 100
        """
        for epoch in range(max_iter):
            error = 0; count = 0
            neuron_error = [None for i in range(self.__depth)]
            data_size = training_data.get_size()

            for i in range(data_size):
                count += 1
                inputs = training_data.get_instance(i)
                target = training_data.get_target(i)

                # FORWARD PROPAGATION
                output = self.forward(inputs)

                # CALCULATE TOTAL ERROR
                correct_idx = 0
                if self.__activation_funcs[self.__depth - 1] == "sfmx":
                    correct_idx = np.where(target == 1)[0][0]
                    error += -np.log(target[correct_idx])
                else:
                    error += (0.5 * (target - output)**2).sum()
                
                # BACKWARD PROPAGATION
                # Calculate delta_output
                output_layer_idx = self.__depth - 1
                act = self.__activation_funcs[output_layer_idx]
                dE_per_do = output - target
                if act != "sfmx":
                    # Calculate dE_per_dnet if activation function is not softmax
                    do_per_dnet = self.__do_per_dnet(self.__sum[output_layer_idx], act)
                    dE_per_dnet = dE_per_do * do_per_dnet
                else:
                    # Calculate dE_per_dnet if activation function is softmax
                    output_idx = np.where(output == output)[0]
                    dE_per_dnet = np.where(output_idx == correct_idx, output - 1, output)
                dnet_per_dw = self.__neuron_output[output_layer_idx - 1]
                dE_per_dw = dE_per_dnet * dnet_per_dw
                
                # Save dE_per_dw
                neuron_error[output_layer_idx] = dE_per_dw
                
                # Calculate delta_hidden for each layer
                for j in range(self.__depth - 2, 0, -1):
                    # Get activation function for this layer
                    act = self.__activation_funcs[j]
                    
                    # Get weight matrix (without bias)
                    weight_matrix = self.get_weights_from_layer(j)[1:]
                    
                    # Calculate sum of dE_per_dh for each hidden unit
                    dEh_per_do = (weight_matrix * neuron_error[j+1][None, :]).sum(axis=1)
                    
                    # Calculate dEh_per_dw for each hidden unit
                    dEh_per_dnet = dEh_per_do * self.__do_per_dnet(self.__sum[j], act)
                    dEh_per_dw = dEh_per_dnet * self.__neuron_output[j-1]
                    
                    # Save dEh_per_dw
                    neuron_error[j] = dEh_per_dw

                # UPDATE WEIGHT
                if count % batch_size == 0 or (count % batch_size != 0 and count == data_size):
                    for a in range(self.__depth - 1):
                        for b in range(self.__neuron_each_layer[a]+1):
                            for c in range(self.__neuron_each_layer[a+1]):
                                self.__weights[a][b][c] -= learning_rate * neuron_error[a+1][c]

            # Break if error less than threshold
            if error < threshold:
                break


    # TODO : implement prediction function
    def predict(self, prediction_data: Tdata) -> np.ndarray:
        """Predict class for each instance in prediction data.
        
        Parameters
        ----------
        `prediction_data` : Tdata,
            Data to predict
        
        Returns
        -------
        ndarray,
            Target class for each instance from prediction data
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
            `../json`
        """
        weights_list = [weight.tolist() for weight in self.__weights]

        neural = {
            "depth" : self.__depth,
            "neuron_each_layer" : self.__neuron_each_layer,
            "weights" : weights_list,
            "biases" : self.__biases.tolist(),
            "activation_funcs" : self.__activation_funcs
        }
        
        json_obj = json.dumps(neural, indent=4)
        with open(path + '/' + file_name, 'w') as file:
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
            self.__weights = [np.array(weight) for weight in json_obj['weights']]
            self.__biases = np.array(json_obj['biases'])
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
                node_label = 'x'
            elif i == self.__depth - 1:
                node_label = 'o'
            else:
                node_label = 'h' + str(i)
            
            # Adding nodes to subgraph
            for j in range(self.__neuron_each_layer[i] + 1):
                node = node_label + str(j)
                if j == 0:
                    if node_label == 'o':
                        pass
                    else:
                        sub_graph.node(node, node, shape='rect')
                else:
                    sub_graph.node(node, node, shape='circle')
            
            # Append subgraph to main graph
            nn_model.subgraph(sub_graph)
            
        # Initialize edges
        for i in range(self.__depth - 1):
            node_src = ''
            node_dst = ''
            if i == 0:
                node_src = 'x'
                if self.__depth == 2:
                    node_dst = 'o'
                else:
                    node_dst = 'h' + str(i+1)
            elif i == self.__depth - 2:
                node_src = 'h' + str(i)
                node_dst = 'o'
            else:
                node_src = 'h' + str(i)
                node_dst = 'h' + str(i+1)
            
            # Initialize edge between j-th neuron from i-th layer to k-th neuron in (i+1)-th layer 
            for j in range(self.__neuron_each_layer[i] + 1):
                for k in range(1, self.__neuron_each_layer[i+1] + 1):
                    weight = round(self.__weights[i][j][k - 1], 2)
                    nn_model.edge(node_src + str(j), node_dst + str(k), xlabel=str(weight), minlen='5')
        
        # render image
        imgPath = nn_model.render(filename=file_name, directory='img/', view=view, format='png', cleanup=True)