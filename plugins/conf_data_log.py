""" Airspace metrics plugin. """
import numpy as np
from datetime import datetime
import pandas as pd
import csv
try:
    from collections.abc import Collection
except ImportError:
    # In python <3.3 collections.abc doesn't exist
    from collections import Collection
from bluesky import stack, traf, sim  #, settings, navdb, traf, sim, scr, tools
from bluesky.tools import areafilter, datalog, plotter, geo, TrafficArrays
from bluesky.tools.aero import nm, ft
import bluesky as bs


# Metrics object
metrics_conf = None


def init_plugin():

    # Addtional initilisation code
    global metrics_conf
    metrics_conf = Metrics_conf()
    # Configuration parameters
    config = {
        'plugin_name': 'conf_data_log',
        'plugin_type': 'sim',
        'update_interval': 2.5,
        'update': metrics_conf.update,
        'reset': metrics_conf.reset
        }

    stackfunctions = {
        'CONF_DATA_METRICS': [
            'CONF_DATA_METRIC ADDSECTOR name',
            'txt,txt',
            metrics_conf.stackio,
            'Print something to the bluesky console based on the flag passed to MYFUN.']
    }

    return config, stackfunctions

class SectorData:
    def __init__(self):
        self.acid = list()
        self.lat0 = np.array([])
        self.lon0 = np.array([])
        self.dist0 = np.array([])

    def id2idx(self, acid):
        # Fast way of finding indices of all ACID's in a given list
        tmp = dict((v, i) for i, v in enumerate(self.acid))
        return [tmp.get(acidi, -1) for acidi in acid]

    def get(self, acid):
        idx = self.id2idx(acid)
        return self.lat0[idx], self.lon0[idx], self.dist0[idx]

    def delete(self, acid):
        idx = np.sort(self.id2idx(acid))
        self.lat0 = np.delete(self.lat0, idx)
        self.lon0 = np.delete(self.lon0, idx)
        self.dist0 = np.delete(self.dist0, idx)
        for i in reversed(idx):
            del self.acid[i]

    def extend(self, acid, lat0, lon0, dist0):
        self.lat0 = np.append(self.lat0, lat0)
        self.lon0 = np.append(self.lon0, lon0)
        self.dist0 = np.append(self.dist0, dist0)
        self.acid.extend(acid)


class Metrics_conf(TrafficArrays):
    def __init__(self):
        super().__init__()

        self.sectors =list()
        # List of sets of aircraft in each sector
        self.acinside = list()
        self.fconfpairs = None
        self.flospairs = None
        self.fdist = None
        self.ftinconfl = None
        # [m] Horizontal separation minimum for detection
        self.rpz = bs.settings.asas_pzr * nm* bs.settings.asas_mar #TK Added bs.settings.asas_mar
        # [m] Vertical separation minimum for detection
        self.hpz = bs.settings.asas_pzh * ft
        # [s] lookahead time
        self.dtlookahead = bs.settings.asas_dtlookahead
    def update(self):

        if not traf.ntraf or not self.sectors:
            return

        # Check convergence using CD with large RPZ and tlook
        confpairs, lospairs, inconf, tcpamax, qdr, dist, dcpa, tcpa, tinconfl, dalt = \
            traf.cd.detect(traf, traf, self.rpz, self.hpz, self.dtlookahead)  # traf.asas.dh

        self.fconfpairs_w.writerow(np.append(sim.simt, confpairs))
        self.flospairs_w.writerow(np.append(sim.simt, lospairs))
        self.fdist_w.writerow(np.append(sim.simt, dist))
        self.ftinconfl_w.writerow(np.append(sim.simt, tinconfl))

        #if sim.simt > 9000 and sim.simt < 9005:
        #    self.fconfpairs.close()
        #    self.flospairs.close()
        #    self.fdist.close()

    def reset(self):
        if self.fconfpairs:
            self.fconfpairs.close()
            self.fconfpairs = None
        if self.flospairs:
            self.flospairs.close()
            self.flospairs = None
        if self.fdist:
            self.fdist.close()
            self.fdist = None
        if self.sectors:
            self.sectors = list()
        if self.acinside:
            self.acinside = list()

    def stackio(self, cmd, name):
        if cmd == 'LIST':
            if not self.sectors:
                return True, 'No registered sectors available'
            else:
                return True, 'Registered sectors:', str.join(', ', self.sectors)
        elif cmd == 'ADDSECTOR':
            if name == 'ALL':
                for name in areafilter.areas.keys():
                    self.stackio('ADDSECTOR', name)
            # Add new sector to list.
            #elif name in self.sectors:
            #    return False, 'Sector %s already registered.' % name
            elif areafilter.hasArea(name):
                if self.fconfpairs:
                    self.fconfpairs.close()
                    self.fconfpairs = None
                if self.flospairs:
                    self.flospairs.close()
                    self.flospairs = None
                if self.fdist:
                    self.fdist.close()
                    self.fdist = None
                if self.sectors:
                    self.sectors = list()
                if self.acinside:
                    self.acinside = list()
                if self.ftinconfl:
                    self.ftinconfl.close()
                    self.ftinconfl = None


                timestamp = datetime.now().strftime('%Y%m%d_%H-%M-%S')
                scenname = stack.get_scenname()

                self.fconfpairs = open('output/fconfpairs_'+scenname+timestamp+'_.csv', 'w', newline='')
                self.flospairs = open('output/flospairs_'+scenname+timestamp+'_.csv', 'w', newline='')
                self.fdist = open('output/fdist_'+scenname+timestamp+'_.csv', 'w', newline='')
                self.ftinconfl = open('output/ftinconfl_'+scenname+timestamp+'_.csv', 'w', newline='')
                self.fconfpairs_w = csv.writer(self.fconfpairs)
                self.flospairs_w = csv.writer(self.flospairs)
                self.fdist_w = csv.writer(self.fdist)
                self.ftinconfl_w = csv.writer(self.ftinconfl )

                print('file_cre')
                # Add new area to the sector list, and add an initial inside count of traffic
                self.sectors.append(name)
                self.acinside.append(SectorData())
                #plotter.legend(self.sectors, 1)
                return True, 'Added %s to sector list.' % name

            else:
                return False, "No area found with name '%s', create it first with one of the shape commands" % name

        else:
            # Remove area from sector list
            if name in self.sectors:
                idx = self.sectors.index(name)
                self.sectors.pop(idx)
                return True, 'Removed %s from sector list.' % name
            else:
                return False, "No sector registered with name '%s'." % name
