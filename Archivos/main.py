# SE IMPORTA LIBRERIAS Y SE ABRE EL ARCHIVO
import pandas as pd

excel = pd.read_excel('Datos.xlsx', sheet_name="Hoja1")
data_frame = pd.DataFrame(excel)

# SE REALIZA LA CLASIFICACION

# SE REALIZA LA IMPORTACION DEL DICCIONARIO TOKENIZADO Y DEL MODELO PREDICTIVO ENTRENADO
import joblib

modelo_predictivo = joblib.load('LR_ent.pkl')
diccionario = joblib.load('Voc.pkl')

#FUNCIONES DE CLASIFICACION

# TOKENIZA, CLASIFICA Y DEVUELVE LA CLASIFICACION
def Clasificar (dato):
    dato_tokenizado = diccionario.transform([dato])
    return modelo_predictivo.predict(dato_tokenizado)

# ESTA FUNCION LA MANIPULACION DE LOS DATOS.
# TOMA EL SET DE DATOS Y LAS COLUMNAS NECESARIAS Y ENTREGA EL SET DE DATOS CON LA CLASIFICACION REALIZADA
def SelectAndConvert(dataset,ColumData, ColumnClas1, ColumnClas2):

    i, a = 0, 0

    while i < len(dataset):
            if pd.isna(dataset[ColumnClas1][i]) or pd.isna(dataset[ColumnClas2][i]):
                dato_a_analizar = dataset[ColumData][i]
                dato_clasificado = Clasificar(dato_a_analizar)
                dato_clasificado_separado = dato_clasificado[0].split()
                dataset[ColumnClas1][i] = dato_clasificado_separado[0].capitalize()
                dataset[ColumnClas2][i] = (' '.join(dato_clasificado_separado[1:len(dato_clasificado_separado)])).capitalize()
                a +=1

            i +=1
    print ("Clasificaciones: ", a)

# SE EJECUTA LA FUNCION
SelectAndConvert(data_frame,'DESCRIPCION DEL MATERIAL','Clasificacion', '2 Clas')

# SE GUARDA EN UN NUEVO ARCHIVO DE EXCEL PARA SU POSTERIOR ANALISIS
from pandas import ExcelWriter
writer = ExcelWriter('Clasificación.xlsx')
data_frame.to_excel(writer, 'Clasificación', index=False)
writer.save()
