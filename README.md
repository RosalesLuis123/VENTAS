Dashboard de Ventas con Predicción y Análisis
Este proyecto es un dashboard interactivo desarrollado con Streamlit que permite visualizar y analizar datos de ventas, así como hacer predicciones utilizando modelos de regresión lineal. El dashboard incluye diversas funcionalidades, como la visualización de ventas por fecha, predicciones de ventas futuras y análisis de ventas por categoría.

Contenido del Proyecto
Visualización de Datos de Ventas: Muestra los primeros registros de las ventas y estadísticas descriptivas.
Gráficos de Ventas: Visualiza las ventas a lo largo del tiempo, agrupadas por fecha.
Predicción de Ventas: Utiliza un modelo de regresión lineal para predecir las ventas futuras.
Análisis de Ventas por Categoría: Muestra las ventas agrupadas por categoría tanto en cantidad como en valor total.
Tecnologías Usadas
Python: Lenguaje de programación utilizado para todo el análisis y desarrollo del dashboard.
Streamlit: Framework utilizado para crear el dashboard interactivo.
Pandas: Biblioteca para manipulación y análisis de datos.
Matplotlib & Seaborn: Bibliotecas para la visualización de datos.
Scikit-learn: Biblioteca utilizada para la implementación de la regresión lineal y las predicciones.
CSV: Formato de archivo utilizado para cargar los datos de ventas.
Instalación
Requisitos Previos
Asegúrate de tener Python 3.x instalado en tu sistema. Si no tienes las bibliotecas necesarias, puedes instalarlas ejecutando el siguiente comando:

bash
Copiar
Editar
pip install streamlit pandas matplotlib seaborn scikit-learn
Ejecución
Clona o descarga este repositorio en tu máquina local.
Asegúrate de tener el archivo CSV de ventas en el mismo directorio que el script.
Para ejecutar el dashboard, abre una terminal y navega al directorio donde se encuentra el archivo de Python (por ejemplo, dashboard.py).
Ejecuta el siguiente comando:
bash
Copiar
Editar
streamlit run dashboard.py
Esto abrirá el dashboard en tu navegador.

Funcionalidades
1. Visualización de Ventas por Fecha
El dashboard muestra las ventas a lo largo del tiempo, agrupadas por fecha. Este gráfico permite ver cómo han fluctuado las ventas y detectar patrones estacionales o tendencias en los datos.

2. Predicción de Ventas Futuras
Utilizando un modelo de regresión lineal de Scikit-learn, el dashboard puede predecir las ventas futuras en función de las fechas pasadas. Los usuarios pueden elegir predecir las ventas para el próximo mes, los próximos dos meses o tres meses.

¿Cómo se hace?
Se utiliza la columna de Fecha y se convierte en un número que representa los días desde el 1 de enero de 1970.
El modelo de regresión lineal entrena con estos datos históricos y luego predice las ventas futuras para una fecha seleccionada.
3. Análisis de Ventas por Categoría
El dashboard también permite visualizar las ventas agrupadas por categoría, mostrando tanto la cantidad de ventas como el total de ventas (valor en dinero generado por esas ventas).

¿Cómo se hace?
Los datos se agrupan por la columna Categoría.
Se calcula la suma de las ventas tanto en cantidad (Cantidad) como en total de ventas (Total Venta).
Los resultados se visualizan en un gráfico combinado con barras para la cantidad de ventas y una línea para el total de ventas.
Estructura del Proyecto
bash
Copiar
Editar
.
├── dashboard.py           # Código del Dashboard en Streamlit
├── ventas.csv             # Archivo CSV con los datos de ventas
├── README.md              # Este archivo
└── Ventas.ipynb           # Archivo un poco explicativo sobre las funciones y librerias expliacadas por separado
Archivo CSV - ventas.csv
Este archivo contiene los datos de ventas que son utilizados por el dashboard. Debe tener las siguientes columnas:

Fecha: Fecha de la venta.
Cantidad: Número de unidades vendidas.
Total Venta: Valor total de las ventas.
Categoría: La categoría a la que pertenece el producto.
Modelo de Predicción
El modelo de predicción se entrena con los datos de ventas anteriores y utiliza regresión lineal para prever las ventas en el futuro.   El modelo está basado en la columna Fecha y la etiqueta Cantidad.

Posibles Mejoras
Implementar predicciones usando otros algoritmos de Machine Learning más avanzados como Random Forest o XGBoost.
Agregar más filtros interactivos para segmentar las ventas por otras características como producto o región.
Añadir más métricas de desempeño del modelo, como el R^2.
Contribuciones
Si deseas contribuir a este proyecto, por favor, crea un fork del repositorio y abre un pull request con tus cambios.

Licencia
Este proyecto está bajo la licencia MIT.
