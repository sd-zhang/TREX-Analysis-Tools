import os
from os import walk
import dataset

def get_filenames_in_path(path):
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
        break
    return f

def extract_data(input_file):
    with open(input_file, 'r') as f:
        # headers = []
        data = []

        # read first 3 lines
        bk_v = f.readline().strip().rstrip('\n')
        headers = f.readline().lower().strip().rstrip('\n').replace('"','').split(',')

        grid_col = headers.index('grid')
        solar_col = headers.index('solar')
        solarP_col = headers.index('solar+')

        # print(headers.index('rid'))

        agg = f.readline().strip().rstrip('\n').split(',')
        t_stop = int(agg[0], 16)
        t_end = int(agg[0], 16)
        # g_end = int(agg[1], 16)
        # s_end = int(agg[2], 16)
        # sp_end = int(agg[3], 16)

        # read data
        while True:
            line = f.readline().strip().rstrip('\n').split(',')
            if not line[0]:
                break
            t_delta = int(line[0], 16)
            t_steps = int(t_delta/60)
            # print(t_step)

            # get raw data. Convert from Ws to Wh
            grid = int(line[grid_col])/3600
            solar = int(line[solar_col]) / 3600
            solarP = int(line[solarP_col]) / 3600

            # TODO: divide and stuff for t_step > 1
            # do not divide and stuff for t_step > 1
            # let the agent figure out how to interpolate
            for step in range(t_steps):
                data_row = {'tstamp': t_end - step * 60,
                            'grid': -grid / t_steps,
                            'solar': -solar / t_steps,
                            'solar+': -solarP / t_steps}
                data.append(data_row)
            t_end -= t_delta

        headers = ['tstamp', 'grid', 'solar', 'solar+']
        return True, headers, data

def egauge_to_sqlite(input_file, output_path=''):
    success, headers, data = extract_data(input_file)
    if success:
        db = dataset.connect('sqlite:///' + input_file.replace('.egauge', '.db'))
        table = db.create_table('profiles', primary_id='tstamp', primary_type=db.types.integer)
        for header in headers:
            table.create_column(header, db.types.bigint)
        table.insert_many(data)
        return True, input_file
    return False, input_file

def add_tstop():



    pass

# def add_geo(config_path, db_path, egauge_id):
#     import re
#     config_file = config_path + egauge_id + '.xml'
#
#     with open(config_file, 'r') as f:
#         for line in f:
#             loc = re.findall("<loc>(.*?)</loc>", str(line))
#             db = dataset.connect('sqlite:///' + input_file.replace('.egauge', '.db'))
#             table = db.create_table('location')
#
#             if loc:
#                 loc = loc[0].split(',')
#                 location = [float(coord) for coord in loc]
#
#                 if location[0] == 0 and location[1] == 0:
#                     return False
#
#                 lats = [dd_start[0], dd_end[0]]
#                 longs = [dd_start[1], dd_end[1]]
#                 lats.sort()
#                 longs.sort()
#
#                 # print(location, lats, longs)
#
#                 if (lats[0] <= location[0] <= lats[1]) and (longs[0] <= location[1] <= longs[1]):
#                     return True
#                 return False
#         return False


# _, headers, data = extract_data('z:/Seattle/egauge13830.egauge')
# egauge_to_sqlite('z:/Seattle/egauge13830.egauge')

f = get_filenames_in_path('z:/Seattle')
new_fs = []
for file in f:
    try:
        new_f = egauge_to_sqlite('z:/Seattle/'+file)
        new_fs.append([1])
    except:
        continue