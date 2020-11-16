import numpy as np
from datetime import datetime
import pandas as pd, csv
try:
    from collections.abc import Collection
except ImportError:
    from collections import Collection

from bluesky import stack, traf, sim
from bluesky.tools import areafilter, datalog, plotter, geo, TrafficArrays
from bluesky.tools.aero import nm, ft
import bluesky as bs
metrics_conf = None

def init_plugin():
    global metrics_conf
    metrics_conf = Metrics_conf()
    config = {'plugin_name':'conf_data_log',
     'plugin_type':'sim',
     'update_interval':2.5,
     'update':metrics_conf.update,
     'reset':metrics_conf.reset}
    stackfunctions = {'CONF_DATA_METRICS': [
                           'CONF_DATA_METRIC ADDSECTOR name',
                           'txt,txt',
                           metrics_conf.stackio,
                           'Print something to the bluesky console based on the flag passed to MYFUN.']}
    return (
     config, stackfunctions)


class SectorData:

    def __init__(self):
        self.acid = list()
        self.lat0 = np.array([])
        self.lon0 = np.array([])
        self.dist0 = np.array([])

    def id2idx(self, acid):
        tmp = dict(((v, i) for i, v in enumerate(self.acid)))
        return [tmp.get(acidi, -1) for acidi in acid]

    def get(self, acid):
        idx = self.id2idx(acid)
        return (self.lat0[idx], self.lon0[idx], self.dist0[idx])

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
        self.sectors = list()
        self.acinside = list()
        self.fconfpairs = None
        self.flospairs = None
        self.fdist = None
        self.ftinconfl = None
        self.rpz = bs.settings.asas_pzr * nm * bs.settings.asas_mar
        self.hpz = bs.settings.asas_pzh * ft
        self.dtlookahead = bs.settings.asas_dtlookahead

    def update(self):
        return traf.ntraf and self.sectors or None
        confpairs, lospairs, inconf, tcpamax, qdr, dist, dcpa, tcpa, tinconfl, dalt = traf.cd.detect(traf, traf, self.rpz, self.hpz, self.dtlookahead)
        self.fconfpairs_w.writerow(np.append(sim.simt, confpairs))
        self.flospairs_w.writerow(np.append(sim.simt, lospairs))
        self.fdist_w.writerow(np.append(sim.simt, dist))
        self.ftinconfl_w.writerow(np.append(sim.simt, tinconfl))

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
                return (True, 'No registered sectors available')
            return (True, 'Registered sectors:', str.join(', ', self.sectors))
        else:
            if cmd == 'ADDSECTOR':
                if name == 'ALL':
                    for name in areafilter.areas.keys():
                        self.stackio('ADDSECTOR', name)

                else:
                    if areafilter.hasArea(name):
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
                        self.fconfpairs = open(('output/fconfpairs_' + scenname + timestamp + '_.csv'), 'w', newline='')
                        self.flospairs = open(('output/flospairs_' + scenname + timestamp + '_.csv'), 'w', newline='')
                        self.fdist = open(('output/fdist_' + scenname + timestamp + '_.csv'), 'w', newline='')
                        self.ftinconfl = open(('output/ftinconfl_' + scenname + timestamp + '_.csv'), 'w', newline='')
                        self.fconfpairs_w = csv.writer(self.fconfpairs)
                        self.flospairs_w = csv.writer(self.flospairs)
                        self.fdist_w = csv.writer(self.fdist)
                        self.ftinconfl_w = csv.writer(self.ftinconfl)
                        print('file_cre')
                        self.sectors.append(name)
                        self.acinside.append(SectorData())
                        return (
                         True, 'Added %s to sector list.' % name)
                    return (
                     False, "No area found with name '%s', create it first with one of the shape commands" % name)
            else:
                if name in self.sectors:
                    idx = self.sectors.index(name)
                    self.sectors.pop(idx)
                    return (True, 'Removed %s from sector list.' % name)
                return (False, "No sector registered with name '%s'." % name)