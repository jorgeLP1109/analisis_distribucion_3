import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.preprocessing import LabelEncoder


datos = pd.read_csv('datos.csv')

# Eliminar columnas no deseadas
columnas_no_deseadas = ['muerto']
# Asegúrate de que las columnas existan antes de intentar eliminarlas
if all(col in datos.columns for col in columnas_no_deseadas):
    # Convertir la columna 'sexo' a números usando Label Encoding
    le = LabelEncoder()
    datos['sexo'] = le.fit_transform(datos['sexo'])
    
    # Rellenar valores faltantes con la media de la columna
    datos = datos.fillna(datos.mean())

    # Eliminar columnas no deseadas
    X = datos.drop(columns=columnas_no_deseadas)

    # Reducción de dimensionalidad usando t-SNE
    X_embedded = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
    
    # Crear DataFrame para Plotly
    df_plotly = pd.DataFrame({'Dimension_1': X_embedded[:, 0], 'Dimension_2': X_embedded[:, 1], 'Dimension_3': X_embedded[:, 2], 'muerto': datos['muerto']})

    # Grafico de dispersión 3D con Plotly
    fig = px.scatter_3d(df_plotly, x='Dimension_1', y='Dimension_2', z='Dimension_3', color='muerto',
                        title='Visualización 3D con t-SNE', labels={'Dimension_1': 'Dimensión 1', 'Dimension_2': 'Dimensión 2', 'Dimension_3': 'Dimensión 3'},
                        color_discrete_map={0: 'green', 1: 'red'})
    fig.show()
else:
    print("Al menos una de las columnas no existe en el DataFrame.")
