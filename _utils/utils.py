import datetime
import gzip
import pickle
import random
from datetime import datetime
from pathlib import Path
import os

import pytz
from dateutil.parser import parse as timeparse

secure_random = random.SystemRandom()

def timestamp_to_local(epoch_ts, timezone):
    """Converts UNIX timestamp to local datetime object"""
    return datetime.fromtimestamp(epoch_ts, pytz.timezone(timezone))

def timestr_to_timestamp(time_string:str, timezone:str):
    timestamp = pytz.timezone(timezone).localize(timeparse(time_string))
    return int(timestamp.timestamp())

def process_profile(row, gen_scale=1, load_scale=1):
    """
        Converts a row of data from energy profile database to format usable by TREX

        Energy profile database is formatted based on a commonly found format by eGauge.
        Most notably, the following four columns exist in some form:
        tstamp, grid, solar, solar+
        Whereas tstamp is the UNIX timestamp, grid is the net load, solar and solar+ are the solar generation
        The only difference between solar and solar+ is that sometimes solar values can be slightly negative.
        The cause of this phenomenon is unknown, but the suspected cause may be due to reverse current flow in some
        parts of the measuring setup. Because of this, solar+ is used for calculations.

        Parameters
        ----------
        row : dict
            Must be a dict, or dict-compatible format consisting of at minimum tstamp, grid, and solar+
            tstamp should be a UNIX timestamp
            grid, solar, and solar+ should be in units of Wh
            the row indicates energy readings between the last tstamp and the end of the current tstamp
        gen_scale : float
            It is helpful to scale the raw profiles in for simulations.
            Default scale is 1
        load_scale : float
            It is helpful to scale the raw profiles in for simulations.
            Default scale is 1

        Returns
        -------
        generation:int, consumption:int
            returns generation and consumption as integer values in units of Wh
            The error accumulated is well within acceptable range
            Integer measures should also be compatible with pulse-watt readings from real meters,
            making transition to real system relatively easy and seamless
        """

    if row is not None:
        consumption = int(round(load_scale * (row['grid'] + row['solar+']), 0))
        generation = int(round(gen_scale * row['solar+'], 0))
        return generation, consumption
    return 0, 0

def energy_to_power(generation, consumption, duration=60, net_load=False):
    # convert from energy (Wh) to average power during interval (s) to kW
    duration_hour_fraction = duration / 3600
    gen_avg_kw = generation / duration_hour_fraction / 1000
    load_avg_kw = consumption / duration_hour_fraction / 1000

    if net_load:
        return load_avg_kw - gen_avg_kw
    else:
        return gen_avg_kw, load_avg_kw


def dump_zp(output_directory, filename, object):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print('Saving...')
    start_time = datetime.now()
    file_path = output_directory + '/' + filename + '.zp'

    f = gzip.open(file_path, 'wb')
    pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    if elapsed_time.seconds > 0:
        print('Time:', elapsed_time)


def import_zp(input_directory, filename):
    start_time = datetime.now()
    file_path = input_directory + '/' + filename + '.zp'
    f = gzip.open(file_path, 'rb')
    p_obj = pickle.load(f)
    f.close()

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    if elapsed_time.seconds > 1:
        print('Loading Time:', elapsed_time)

    return p_obj