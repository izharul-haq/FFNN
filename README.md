# FFNN

Simple feedforward neural network implementation using Python and NumPy

## Dependency

1. Matplotlib
   Install matplotlib by typing this command in command prompt/terminal

   ```
   python -m pip install matplotlib
   ```

2. Graphviz
   Download and install graphviz from [here](https://graphviz.org/download/) and add graphviz `bin` path (ex. `C:/Program Files/Graphviz/bin`) to PATH (Windows only). Lastly, install python package for graphviz by typing this command in command prompt/terminal

   ```
   python -m pip install graphviz
   ```

## How to Use

1. Save data in Tdata object or load data from an external file (.json)

   ```python
   data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
   trgt = np.array ([0, 1, 1, 0])
   t_data = Tdata(data, trgt)
   
   # OR
   
   t_data = Tdata()
   t_data.load('data_file.json')
   ```

2. Create a Neural model or load model from an external file (.json)

   ```python
   nn = Neural([2, 2, 1])
   
   # OR
   
   nn = Neural()
   nn.load('model_file.json')
   ```

3. Train model with `fit` method using Tdata object as training data

4. Predict new instance(s) with `predict` method

5. Refer to `test.py` as an example how to use model using iris dataset.
