import pandas as pd
import matplotlib.pyplot as plt

# Lee los datos desde el archivo CSV sin encabezado
df = pd.read_csv('datasets/xor2.csv', header=None, names=['f1', 'f2', 'clase'])

# Extrae coordenadas x, y y la clase
x = df['f1']
y = df['f2']
clase = df['clase']

# Itera sobre las clases únicas y dibuja puntos para cada clase
for c in clase.unique():
    indices = clase == c
    plt.scatter(x[indices], y[indices], label=f'Clase {c}')

# Configura las etiquetas
plt.xlabel('Eje F1')
plt.ylabel('Eje F2')

# Muestra la leyenda
# plt.legend()

# Muestra el gráfico
#plt.show()

# Guarda el gráfico
plt.savefig('datasets/xor2.png')
