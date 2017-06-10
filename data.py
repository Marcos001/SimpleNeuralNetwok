
import numpy as np

def create_data_for_train():
    ''''''
    _vetor_desejado = []
    _vetor_entrada = []
    _matrix = []

    file = open('/home/nig/PycharmProjects/MLP-Recognition/data_vetor.txt', 'r')
    data = file.readlines()
    
    for i in range(len(data)):
        linha = data[i].split('\n')
        key, value = str(linha).split(':')

        key = key.replace("['",'')
        value = value.replace(' ', '')
        value = value.replace("','']", '')
        value = value.replace(".']", '')

        tmp = key, value
        _matrix.append(tmp)

    print('saving data for train-data')
    np.save('train-data.npy', _matrix)
    file.close()

def loading_data_train():
    ''
    return np.load('train-data.npy')

def view_data_for_train():
    ''
    print(loading_data_train())

def create_data_for_test():
    ''


if __name__ == '__main__':
    create_data_for_train()
    view_data_for_train()