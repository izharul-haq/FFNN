from python.Tdata import Tdata
from python.Neural import Neural

t_data = Tdata()
t_data.load('json/data/data_test.json')

nn_model = Neural()
nn_model.load('./json/model/model_test.json')
print(nn_model.forward(t_data.get_instance(0)))
nn_model.fit(t_data, max_iter=10000)
print(nn_model.forward(t_data.get_instance(0)))