import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar los datos
data = pd.read_csv('ingreso.csv')

# Mostrar algunas estadísticas descriptivas
print(data.describe())

# Crear una instancia del modelo de regresión lineal
modelo_regresion = LinearRegression()

# Separar las variables independientes (X) y dependientes (y)
X = data[['horas']]
y = data['ingreso']

# Entrenar el modelo de regresión lineal
modelo_regresion.fit(X, y)

# Obtener el intercepto (b0)
intercepto = modelo_regresion.intercept_

# Obtener el coeficiente (pendiente) (b1)
coeficiente = modelo_regresion.coef_

print(f"Intercepto: {intercepto}")
print(f"Coeficiente: {coeficiente}")

# Graficar los datos de dispersión
plt.scatter(X, y, label='Datos de dispersión')

# Graficar la regresión lineal
plt.plot(X, modelo_regresion.predict(X), color='red', label='Regresión lineal')

# Agregar etiquetas y leyenda
plt.xlabel('Horas de trabajo semanal')
plt.ylabel('Ingreso mensual')
plt.legend()

# Mostrar la gráfica
plt.show()
