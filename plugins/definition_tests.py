
from bluesky import stack, scr

def init_plugin():
    config = {'plugin_name':'def_test',
     'plugin_type':'sim'}
    stackfunctions = {'PRINTING': [
                  'MYFUN ON/OFF',
                  '[onoff]',
                  printing,
                  'Print something to the bluesky console based on the flag passed to MYFUN.']}
    return (
     config, stackfunctions)


def printing(y=True):
    X = 'TEST1'
    return (True, X)