from _utils import utils, db_utils
from sqlalchemy import create_engine, MetaData, Column, func, select
from sqlalchemy.orm import sessionmaker
import dataset
import process_profile

# print("Select AVG (generation) from test_profile_1kw_square_p4+2")
# Select COUNT(cust_code) from customers;
# Select MAX(column_name) from table_name;
# Select MIN(column_name) from table_name;
# Select STDDEV(column_name) from table_name;
#
# print(select())
# db.query('Select AVG generation from test_profile_1kw_square_p4+2')

def calculate_stats(profile_db_location:str, profile_name:str):
    engine = create_engine(profile_db_location)
    table = db_utils.get_table(profile_db_location, profile_name, engine=engine)
    Session = sessionmaker(bind=engine)

    session = Session()
    stats = dict()
    stats['name'] = profile_name
    stats['min_generation'] = float(session.query(func.min(table.c.generation)).scalar())
    stats['max_generation'] = float(session.query(func.max(table.c.generation)).scalar())
    stats['avg_generation'] = float(session.query(func.avg(table.c.generation)).scalar())
    stats['stddev_generation'] = float(session.query(func.stddev(table.c.generation)).scalar())
    stats['min_consumption'] = float(session.query(func.min(table.c.consumption)).scalar())
    stats['max_consumption'] = float(session.query(func.max(table.c.consumption)).scalar())
    stats['avg_consumption'] = float(session.query(func.avg(table.c.consumption)).scalar())
    stats['stddev_consumption'] = float(session.query(func.stddev(table.c.consumption)).scalar())
    session.close()
    return stats

def store_stats(profile_db_location, data):
    db = dataset.connect(profile_db_location)
    table = db.get_table('_statistics', primary_id='name', primary_type=db.types.text)
    table.upsert(data, ['name'])

if __name__ == "__main__":
    profile_db_location = "postgresql://postgres:postgres@localhost/profiles"
    profile_name = "test_profile_1kw_square_p2+1"
    stats = calculate_stats(profile_db_location, profile_name)
    store_stats(profile_db_location, stats)
