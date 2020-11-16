# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:28:51 2020

@author: thoma
"""
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, scr #, settings, navdb, traf, sim, scr, tools

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
# Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'def_test',

        # The type of this plugin.
        'plugin_type':     'sim'
        }

    stackfunctions = {
        # The command name for your function
        'PRINTING': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'MYFUN ON/OFF',

            # A list of the argument types your function accepts.
            '[onoff]',

            # The name of your function in this plugin
            printing,

            # a longer help text of your function.
            'Print something to the bluesky console based on the flag passed to MYFUN.']
            
            }

    return config, stackfunctions


def printing(y=True):
    #scr.echo('test')
    X = 'TEST1'
    return True, X


