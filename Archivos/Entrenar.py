import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# LEO ARCHIVO DE DATOS
excel = pd.read_excel('Datos.xlsx', sheet_name="Hoja1")

# LIMPIO LAS FILAS SIN DATOS (N/A)

excel = excel.dropna()

# CREO LAS VARIABLES PARA EL ENTRENAMIENTO
dfX = excel['DESCRIPCION DEL MATERIAL'].str.lower()
dfy1 = excel['Clasificacion'].str.lower()
dfy2 = excel['2 Clas'].str.lower()
dfy = (dfy1 + " " + dfy2)

# SE TOKENIZAN Y VECTORIZAN LOS DATOS A CLASIFICAR Y LAS CLASIFICACIONES
vector = CountVectorizer()
VF = vector.fit(dfX)
# print(vector.vocabulary_)

BX = vector.transform(dfX)
BY = dfy

# SE SEPARAN LOS DATOS EN DATOS DE ENTRENAMIENTO Y DATOS DE PRUEBA
X_ent, X_test, Y_ent, Y_test = train_test_split(BX, BY)

# SE CARGA EL MODELO Y SE ENTRENA
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
CLF = lr.fit(X_ent, Y_ent)
Score = (CLF.score(X_test, Y_test))*100

print("El % predictivo del modelo es del {:,.2f} %". format(Score))

# print(lr.predict(X_test))

# SE EXPORTA EL MODELO Y EL DICCIONARIO TOKENIZADO Y SE GUARDA EN ARCHIVOS PKL
import joblib

# MODELO
joblib.dump(CLF, 'LR_ent.pkl')

#DICCIONARIO
joblib.dump(VF, 'Voc.pkl')