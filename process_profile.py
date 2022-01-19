import dataset
from _utils import utils


def process_profile(profile_db_location:str, profile_name:str):
    db = dataset.connect(profile_db_location)
    table = db[profile_name]
    # table.drop_column('generation')
    # table.drop_column('consumption')
    if not table.has_column('generation'):
        table.create_column('generation', type=db.types.integer)

    if not table.has_column('consumption'):
        table.create_column('consumption', type=db.types.integer)
    # db_out = dataset.connect(profiles_processed_location)
    # table_out = db_out.create_table(profile_name, primary_id='time', primary_increment=False)

    for row in table:
        generation, consumption = utils.process_profile(row)
        new_row = {
            'tstamp': row['tstamp'],
            'generation': generation,
            'consumption': consumption
        }
        table.update(new_row, ['tstamp'])


if __name__ == "__main__":
    profile_db_location = "postgresql://postgres:postgres@localhost/profiles"
    profile_name = "test_profile_1kw_square_p4+2"
    process_profile(profile_db_location, profile_name)
