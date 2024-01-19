import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt


# Función para cargar el dataset generado
def cargar_dataset():
    df = pd.read_csv('datasets/2entradas.csv', header=None, names=['f1', 'f2', 'clase'])
    return df

# Función para visualizar el conjunto de datos y la recta del perceptrón
def visualizar(dataset, pesos, sesgo):
    # Extrae coordenadas x, y y la clase
    x = dataset['f1']
    y = dataset['f2']
    clase = dataset['clase']

    # Itera sobre las clases únicas y dibuja puntos para cada clase
    for c in clase.unique():
        indices = clase == c
        plt.scatter(x[indices], y[indices], label=f'Clase {c}')

    x_range = np.linspace(min(dataset['f1']), max(dataset['f1']), 500)
    y_range = np.linspace(min(dataset['f2']), max(dataset['f2']), 500)

    x1, x2 = np.meshgrid(x_range, y_range)
    y = pesos[0] * x1 + pesos[1] * x2 + sesgo

    plt.contour(x1, x2, y, levels=[0], colors='black')
    
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.legend()
    plt.title('Conjunto de Datos y Recta del Perceptron')
    #plt.show()
    # Guardar la figura
    plt.savefig('datasets/2entradas.png')


# Cargar el dataset
dataset = cargar_dataset()

# Pesos y sesgo del perceptrón entrenado
pesos = np.array([-45.0279, 18.7887 ])
sesgo = 2 
 


# Visualizar el conjunto de datos y la recta del perceptrón
visualizar(dataset, pesos, sesgo)
