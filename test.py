from python.Tdata import Tdata
from python.Neural import Neural
from python.utils import convert_label as cl
from python.utils import scoring as scr
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Load raw training and testing data
df = load_iris()
x_train, x_test, y_train, y_test = train_test_split(df.data, df.target, test_size=0.1, random_state=42)

# Initialize Tdata instance for training and testing
train_data = Tdata(x_train, cl.unilabel_to_multilabel(y_train, 3))
test_data = Tdata(x_test, cl.unilabel_to_multilabel(y_test, 3))

# Split training data into 9 equal size
equal_size = len(df.data) // 10
train_data_list = []
for i in range(9):
    start_idx = i * equal_size
    train_data_list.append(train_data.get_some(start_idx, start_idx + equal_size))

# Initialize neural network model
nn_model = Neural()
nn_model.load('./json/model/model_iris.json')

# Train model
for training_data in train_data_list:
    nn_model.fit(training_data, max_iter=1000)

# Save model (.png) after learning
nn_model.show('model_iris')

# Predict testing model
pred = nn_model.predict(test_data.get_instances())
pred = cl.multilabel_to_unilabel(pred)

# Create confusion matrix
conf_matrix = scr.create_conf_matrix(y_test, pred, 3)
print('Confusion matrix:')
print(conf_matrix)

# Print metric for this model
print(f'accuracy score: {scr.accuracy(conf_matrix)}')
print(f'precision score for each class: {scr.precision(conf_matrix)}')
print(f'recall score for each class: {scr.recall(conf_matrix)}')
print(f'f1 score for each class: {scr.f1(conf_matrix)}')