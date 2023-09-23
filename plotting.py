import logging
from sqlalchemy.engine import URL
from sqlalchemy.sql import text
import coloredlogs
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #CARGAR VARIABLES DE CONECION DESDE .env
    load_dotenv()
    url_object = URL.create(
        "postgresql",
        username=os.getenv('USER'),
        password=os.getenv('PASS'),  # plain (unescaped) text
        host=os.getenv('HOST'),
        database=os.getenv('SCHEMA'),
    )
    #crear engine
    engine = create_engine(url_object)
    #conectarse
    conn = engine.connect()

    #logger.info("")
    #logger.info("# 0. RawQuery")

    #aqui hacer query en formato postgreSQL
    my_query = text('SELECT * FROM "AgileBuyings" LIMIT 1')
    resultSet = conn.execute(my_query).fetchall()
    #de auqi en adelante jugar con los resultados de la query
    print(resultSet)
    #cerrar conexion
    conn.close()