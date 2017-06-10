#http://scikit-learn.org/stable/modules/neural_networks_supervised.html#multi-layer-perceptron
#http://scikit-learn.org/stable/modules/neural_networks_supervised.html

from sklearn.neural_network import MLPClassifier
from data import loading_data_train

x_train = loading_data_train()

print('loading data > ')
#X = [[0., 0.], [1., 1.]]      #-> array X of size (n_samples, n_features)
#y = [0, 1]                    #-> array y of size (n_samples,)

matrix = loading_data_train()

y = []
X = []
vetor_x = []

for i in matrix:
    lis_tmp = []
    try:
        value = float((i[1]))
        y.append(int(i[0]))
        lis_tmp.append((i[1]))
        X.append(lis_tmp)
        vetor_x.append((y, X))
    except:
        value = float((i[1]))
        lis_tmp.append(value)
        X.append(lis_tmp)
        y.append(int(ord(i[0])))
    #

print(len(y),' # ',  y)
print(len(X), ' # ', X)

#X = x_train                    #-> array X of size (n_samples, n_features) -> entrada de treino - dados de entrada
#y = y_train                    #-> array y of size (n_samples,)            -> saida desejada - valor alvo

#"""
print('create model > ')
_network = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

"""
_network = MLPClassifier(activation='relu', 
                         alpha=1e-05, 
                         batch_size='auto',
                         early_stopping=False, 
                         epsilon=1e-08, 
                         hidden_layer_sizes=(5, 2), 
                         learning_rate='constant', 
                         learning_rate_init=0.001, 
                         max_iter=200, 
                         momentum=0.9, 
                         nesterovs_momentum=True, 
                         power_t=0.5, 
                         random_state=1, 
                         shuffle=True, 
                         solver='lbfgs', 
                         tol=0.0001, 
                         validation_fraction=0.1, 
                         verbose=True, 
                         warm_start=False)

"""

print('training model > ')
_network.fit(X, y)
print('finish training.')

print('fazendo predição dos dados > ', _network.predict(1))
