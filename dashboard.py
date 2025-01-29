import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Título del dashboard
st.title("Dashboard de Ventas")

# Cargar datos desde tu archivo CSV
df = pd.read_csv("ventas.csv")

# Mostrar los primeros 5 registros
st.write("Primeras 5 filas de datos", df.head())

# Asegúrate de que la columna 'Fecha' esté en formato de fecha
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Mostrar estadísticas descriptivas
st.write("Estadísticas descriptivas", df.describe())

# Gráfico de ventas por fecha
st.subheader("Ventas por Fecha")
ventas_por_fecha = df.groupby('Fecha')['Cantidad'].sum()

# Gráfico de líneas de ventas por fecha
plt.figure(figsize=(10,6))
plt.plot(ventas_por_fecha.index, ventas_por_fecha.values)
plt.title("Ventas por Fecha")
plt.xlabel("Fecha")
plt.ylabel("Cantidad de Ventas")
st.pyplot(plt)

# Predicciones con Scikit-learn
st.subheader("Predicción de Ventas a Futuro con Scikit-learn")

# Convertir las fechas a números para que el modelo los entienda
df['Fecha_num'] = df['Fecha'].apply(lambda x: (x - pd.Timestamp("1970-01-01")).days)

# Características (X) y Etiqueta (y)
X = df[['Fecha_num']]  # Características (fecha en días)
y = df['Cantidad']  # Etiqueta (ventas)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Mostrar el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
st.write(f"Error cuadrático medio (MSE): {mse:.2f}")

# Realizar predicciones en todas las fechas disponibles
predictions = model.predict(X)

# Graficar las predicciones junto con las ventas reales
plt.figure(figsize=(10,6))
plt.plot(df['Fecha'], df['Cantidad'], label='Ventas Reales', color='blue')
plt.plot(df['Fecha'], predictions, label='Predicción de Ventas', linestyle='--', color='red')
plt.title("Predicción de Ventas con Regresión Lineal")
plt.xlabel("Fecha")
plt.ylabel("Cantidad de Ventas")
plt.legend()
st.pyplot(plt)

# Mostrar el modelo y sus coeficientes
st.write("Coeficiente del modelo:", model.coef_)
st.write("Intercepto del modelo:", model.intercept_)

# Formulario para elegir el número de meses para la predicción
st.subheader("Predicción de Ventas para el Futuro")

# Selección de número de meses para predecir
meses_a_predecir = st.selectbox("Selecciona el número de meses para predecir", [1, 2, 3])

# Obtener la fecha más reciente de los datos
fecha_max = df['Fecha'].max()

# Calcular la fecha de predicción según el número de meses seleccionados
fecha_prediccion = fecha_max + pd.DateOffset(months=meses_a_predecir)

# Convertir la fecha de predicción a número de días
fecha_prediccion_num = (fecha_prediccion - pd.Timestamp("1970-01-01")).days

# Realizar la predicción para la fecha seleccionada
prediccion_ventas = model.predict([[fecha_prediccion_num]])

# Mostrar la predicción en el dashboard
st.write(f"Predicción de ventas para {fecha_prediccion.strftime('%B')} ({meses_a_predecir} mes(es) desde la última fecha): {prediccion_ventas[0]:.2f}")

# Visualizar ventas por categoría: Cantidad de ventas y Total de ventas
st.subheader("Ventas por Categoría (Cantidad y Total Venta)")

# Verifica si tienes una columna llamada 'Categoría', si no, ajusta el nombre
if 'Categoría' in df.columns and 'Total Venta' in df.columns:
    # Agrupar por categoría y obtener la suma de la cantidad de ventas y el total de ventas
    ventas_por_categoria = df.groupby('Categoría').agg(
        cantidad_ventas=('Cantidad', 'sum'),
        total_ventas=('Total Venta', 'sum')
    ).sort_values(by='cantidad_ventas', ascending=False)

    # Mostrar los datos de ventas por categoría
    st.write(ventas_por_categoria)

    # Gráfico de barras de ventas por categoría
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Gráfico de barras para la cantidad de ventas
    ax1.bar(ventas_por_categoria.index, ventas_por_categoria['cantidad_ventas'], color='b', alpha=0.6, label='Cantidad de Ventas')

    # Crear un eje secundario para el total de ventas
    ax2 = ax1.twinx()
    ax2.plot(ventas_por_categoria.index, ventas_por_categoria['total_ventas'], color='r', marker='o', label='Total de Ventas', linestyle='--')

    ax1.set_xlabel("Categoría")
    ax1.set_ylabel("Cantidad de Ventas", color='b')
    ax2.set_ylabel("Total de Ventas ($)", color='r')
    ax1.set_title("Ventas por Categoría - Cantidad y Total de Venta")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Mostrar el gráfico
    st.pyplot(fig)

else:
    st.write("No se encuentran las columnas 'Categoría' o 'Total Venta' en los datos.")
