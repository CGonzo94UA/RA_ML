import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Crear datos para un problema no linealmente separable
np.random.seed(42)

# Definir el margen
margen = 0.2

# Clase 1: Puntos en el cuadrante superior izquierdo y cuadrante inferior derecho con margen
class1_x = np.concatenate([
    np.random.uniform(-1 + margen, -margen, 300),
    np.random.uniform(margen, 1 - margen, 300)
])
class1_y = np.concatenate([
    np.random.uniform(margen, 1 - margen, 300),
    np.random.uniform(-1 + margen, -margen, 300)
])

# Clase 2: Puntos en el cuadrante superior derecho y cuadrante inferior izquierdo con margen
class2_x = np.concatenate([
    np.random.uniform(margen, 1 - margen, 300),
    np.random.uniform(-1 + margen, -margen, 300)
])
class2_y = np.concatenate([
    np.random.uniform(margen, 1 - margen, 300),
    np.random.uniform(-1 + margen, -margen, 300)
])


# Crear un DataFrame con los datos
data = pd.DataFrame({
    'Feature1': np.concatenate([class1_x, class2_x]),
    'Feature2': np.concatenate([class1_y, class2_y]),
    'Label': np.concatenate([np.ones_like(class1_x), -1*np.ones_like(class2_x)])
})

# Mezclar aleatoriamente las filas del DataFrame
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Guardar el DataFrame en un archivo CSV
data.to_csv('datasets/nonlinear_dataset.csv', index=False, header=False)

# Verificar el DataFrame
print(data.head())

# Plotear el conjunto de datos
plt.scatter(class1_x, class1_y, label='Clase 1')
plt.scatter(class2_x, class2_y, label='Clase 2')
plt.title('Conjunto de Datos No Linealmente Separable')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.show()
