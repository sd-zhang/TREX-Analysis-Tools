import numpy as np
from _utils import utils
import dataset

def ts_and_duration(start_datetime_str, end_datetime_str, timezone):
    start_timestamp = utils.timestr_to_timestamp(start_datetime_str, timezone)
    end_timestamp = utils.timestr_to_timestamp(end_datetime_str, timezone)
    duration_minutes = int((end_timestamp - start_timestamp) / 60)
    timestamps = np.linspace(start_timestamp, end_timestamp, duration_minutes)
    return timestamps, duration_minutes

def generate_flat_profile(start_datetime_str, end_datetime_str, timezone, peak_power):
    # peak power in Watts
    timestamps, duration_minutes = ts_and_duration(start_datetime_str, end_datetime_str, timezone)
    power_profile = peak_power * np.ones(duration_minutes)
    energy_profile = (60/3600) * power_profile
    return timestamps, energy_profile

def generate_cosine_profile(start_datetime_str, end_datetime_str, timezone, peak_power, time_offset=0):
    # peak power in Watts
    # time_offset in minutes, defaults to 0
    # start_datetime = pytz.timezone(timezone).localize(timeparse(start_datetime_str))
    # end_datetime = pytz.timezone(timezone).localize(timeparse(end_datetime_str))
    # start_timestamp = start_datetime.timestamp()
    # end_timestamp = end_datetime.timestamp()
    # duration_minutes = int((end_timestamp - start_timestamp) / 60)
    # timestamps = np.linspace(start_timestamp, end_timestamp, duration_minutes)

    timestamps, duration_minutes = ts_and_duration(start_datetime_str, end_datetime_str, timezone)
    x = np.linspace(0, duration_minutes, duration_minutes)
    power_profile = (peak_power/2) * np.cos((2 * np.pi/1440) * (x + time_offset)) + (peak_power/2)
    energy_profile = (60/3600) * power_profile
    return timestamps, energy_profile

def write_to_db(timestamps, energy_profile, db_str:str, profile_name:str):
    profile = [{
        'tstamp': int(timestamps[idx]),
        'grid': 0,
        'solar': energy_profile[idx],
        'solar+': energy_profile[idx]
        } for idx in range(len(timestamps))]

    db = dataset.connect(db_str)
    db.create_table(profile_name, primary_id='tstamp')
    db[profile_name].insert_many(profile)

start_time = '2000-01-01 0:0:0'
end_time = '2030-01-01 0:0:0'
timezone = 'America/Vancouver'
# time_offset = 0
time_offset = int(1440/2)
# timestamps, energy_profile = generate_cosine_profile(start_time, end_time, timezone, 1000, time_offset=int(1440/2))
timestamps, energy_profile = generate_flat_profile(start_time, end_time, timezone, 1000)

write_to_db(timestamps, energy_profile,
            'postgresql://postgres:postgres@localhost/profiles',
            'test_profile_1kw_flat')

# plt.plot(timestamps, energy_profile)
# plt.show()

