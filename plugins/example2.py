from bluesky import traf, scr
import numpy as np

def init_plugin():
    config = {'plugin_name':'EXAMPLE2',
     'plugin_type':'sim',
     'update_interval':2.5,
     'update':update,
     'preupdate':preupdate,
     'reset':reset}
    stackfunctions = {'MYFUN': [
               'MYFUN ON/OFF',
               '[onoff]',
               myfun,
               'Print something to the bluesky console based on the flag passed to MYFUN.']}
    return (
     config, stackfunctions)


def update():
    scr.echo('Average position of traffic lat/lon is: %.2f, %.2f' % (
     np.average(traf.lat), np.average(traf.lon)))


def preupdate():
    pass


def reset():
    pass


def myfun(flag=True):
    return (
     True, 'My plugin received an o%s flag.' % ('n' if flag else 'ff'))