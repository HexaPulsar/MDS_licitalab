# MDS_licitalab
Repositorio del curso Proyecto de Ciencia de Datos - Proyecto LicitaLab

---



1. Clonar el repo en una ubicación local.

2. Crear virtual environment en la terminal con ```python -m venv <nombre environment>```y activarlo con ```.\<nombre environment>\Scripts\Activate.ps1``` (windows)

3. Para instalar las librerías necesarias ```pip install -r 'requirements.txt'```

4. Abrir el archivo plotting.py y hacer los MEJORES GRÁFICOS del mundo!

5. MUY IMPORTANTE: no commitear al main!


# Instrucciones para lanzar el dash board :D

1. clonar el repo
2. tener el archivo  `20231007200103_query_results.csv` en el directorio /Modeling/
3. tener las librerías necesarias instaladas (no arregle el requirements, pido perdón)
4. Revisar si los settings son adecuados en `Modeling/ProviderDescriptionBased/config/config.yaml` (estan como yo los dejé así que solo habría que cambiar el path del csv, que es el del punto 2)
5. lanzar el archivo `generate_corpus.py`. Genera el csv de los vectores. Tener paciencia porque se demora. No aparece nada pero esta trabajando. Si bajas el numero de rows en el `config.yaml` se demora menos.
6. lanzar el archivo 'generate_kmeans.py' el numero de clusters se genera automaticamente, pero si quisieras puedes entrar al codigo pa elegir un numero de clusters. 
7. Si todo salio bien, lanzar `exploratory_dash.py`. En la consola te saldra un link para ir al entorno dashboard.

