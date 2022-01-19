# https://stackoverflow.com/questions/30778015/how-to-increase-the-max-connections-in-postgres

import sqlalchemy
from sqlalchemy import create_engine, MetaData, Column, func
from sqlalchemy_utils import database_exists, create_database

def create_db(db_string):
    engine = create_engine(db_string)
    if not database_exists(engine.url):
        create_database(engine.url)
    return database_exists(engine.url)

def get_table(db_string, table_name, engine=None):
    if not engine:
        engine = create_engine(db_string)

    if not sqlalchemy.inspect(engine).has_table(table_name):
        return None

    metadata = MetaData()
    table = sqlalchemy.Table(table_name, metadata, autoload=True, autoload_with=engine)
    return table

# def get_table_len(db_string, table):
#     engine = create_engine(db_string)
#     with engine.begin() as connection:
#         rows = connection.execute(func.count(table.id)).scalar()
#     # rows = engine.query(func.count(table.id)).scalar()
#     return rows
#     # return engine.scalar(table.count())