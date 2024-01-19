import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Crear datos para un problema no linealmente separable
np.random.seed(42)

# Definir el margen
margen = 0.5
num_puntos = 400
limite = 6

# Clase 1: Puntos en el cuadrante superior izquierdo y cuadrante inferior derecho con margen
class1_x = np.concatenate([
    np.random.uniform(-limite, -margen, num_puntos),
    np.random.uniform(margen, limite, num_puntos)
])
class1_y = np.concatenate([
    np.random.uniform(margen, limite, num_puntos),
    np.random.uniform(-limite, -margen, num_puntos)
])

# Clase 2: Puntos en el cuadrante superior derecho y cuadrante inferior izquierdo con margen
class2_x = np.concatenate([
    np.random.uniform(margen, limite, num_puntos),
    np.random.uniform(-limite, -margen, num_puntos)
])
class2_y = np.concatenate([
    np.random.uniform(margen, limite, num_puntos),
    np.random.uniform(-limite, -margen, num_puntos)
])

# Redondear a 4 decimales
class1_x = np.round(class1_x, decimals=4)
class1_y = np.round(class1_y, decimals=4)
class2_x = np.round(class2_x, decimals=4)
class2_y = np.round(class2_y, decimals=4)

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
#plt.show()

plt.savefig('datasets/nonlinear_dataset.png')
