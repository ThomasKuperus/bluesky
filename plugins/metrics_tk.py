import numpy as np
try:
    from collections.abc import Collection
except ImportError:
    from collections import Collection

from bluesky import stack, traf, sim
from bluesky.tools import areafilter, datalog, plotter, geo, TrafficArrays
from bluesky.tools.aero import nm, ft
metrics = None

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


class Metrics(TrafficArrays):

    def __init__(self):
        super().__init__()
        self.sectors = list()
        self.acinside = list()
        self.sectorsd = np.array([], dtype=(np.int))
        self.sectorconv = np.array([], dtype=(np.float))
        self.sectoreff = []
        self.effplot = None
        self.delac = SectorData()
        self.fsd = None
        self.fconv = None
        self.feff = None

    def create(self, n=1):
        pass

    def delete(self, idx):
        self.delac.extend(np.array(traf.id)[idx], traf.lat[idx], traf.lon[idx], traf.distflown[idx])

    def update(self):
        self.sectorsd = np.zeros(len(self.sectors))
        self.sectorconv = np.zeros(len(self.sectors))
        self.sectoreff = []
        return traf.ntraf and self.sectors or None
        confpairs, lospairs, inconf, tcpamax, qdr, dist, dcpa, tcpa, tLOS = traf.cd.detect(traf, traf, 20 * nm, 1000, 3600)
        if confpairs:
            own, intr = zip(*confpairs)
            ownidx = traf.id2idx(own)
            mask = traf.alt[ownidx] > 70 * ft
            ownidx = np.array(ownidx)[mask]
            dcpa = np.array(dcpa)[mask]
            tcpa = np.array(tcpa)[mask]
        else:
            ownidx = np.array([])
        sendeff = False
        for idx, (sector, previnside) in enumerate(zip(self.sectors, self.acinside)):
            inside = areafilter.checkInside(sector, traf.lat, traf.lon, traf.alt)
            sectoreff = []
            previds = set(previnside.acid)
            ids = set(np.array(traf.id)[inside])
            arrived = list(ids - previds)
            left = previds - ids
            left_intraf = left.intersection(traf.id)
            left_del = list(left - left_intraf)
            left_intraf = list(left_intraf)
            arridx = traf.id2idx(arrived)
            leftidx = traf.id2idx(left_intraf)
            arrdist = traf.distflown[arridx]
            arrlat = traf.lat[arridx]
            arrlon = traf.lon[arridx]
            leftlat, leftlon, leftdist = self.delac.get(left_del)
            leftlat = np.append(leftlat, traf.lat[leftidx])
            leftlon = np.append(leftlon, traf.lon[leftidx])
            leftdist = np.append(leftdist, traf.distflown[leftidx])
            leftlat0, leftlon0, leftdist0 = previnside.get(left_del + left_intraf)
            self.delac.delete(left_del)
            if len(left) > 0:
                q, d = geo.qdrdist(leftlat0, leftlon0, leftlat, leftlon)
                mask = d > 10
                sectoreff = list((leftdist[mask] - leftdist0[mask]) / d[mask] / nm)
                names = np.array(left_del + left_intraf)[mask]
                for name, eff in zip(names, sectoreff):
                    self.feff.write('{}, {}, {}\n'.format(sim.simt, name, eff))

                sendeff = True
            else:
                previnside.delete(left)
                previnside.extend(arrived, arrlat, arrlon, arrdist)
                self.sectoreff.append(sectoreff)
                self.sectorsd[idx] = np.count_nonzero(inside)
                insidx = np.where(np.logical_and(inside, inconf))
                pairsinside = np.isin(ownidx, insidx)
                if len(pairsinside):
                    tnorm = np.array(tcpa)[pairsinside] / 300.0
                    dcpanorm = np.array(dcpa)[pairsinside] / (5.0 * nm)
                    self.sectorconv[idx] = np.sum(np.sqrt(2.0 / tnorm * tnorm + dcpanorm * dcpanorm))
                else:
                    pass
                self.sectorconv[idx] = 0
            self.fconv.write('{}, {}\n'.format(sim.simt, self.sectorconv[idx]))
            self.fsd.write('{}, {}\n'.format(sim.simt, self.sectorsd[idx]))
            self.fconflict.write('{}, {}\n'.format(confpairs, lospairs, inconf, tcpamax, qdr, dist, dcpa, tcpa, tLOS))

        if sendeff:
            self.effplot.send()

    def reset(self):
        if self.fconv:
            self.fconv.close()
        if self.fsd:
            self.fsd.close()
        if self.feff:
            self.feff.close()

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
                    if name in self.sectors:
                        return (
                         False, 'Sector %s already registered.' % name)
                    if areafilter.hasArea(name):
                        if not self.sectors:
                            self.fconv = open('output/convergence.csv', 'w')
                            self.fsd = open('output/density.csv', 'w')
                            self.feff = open('output/efficiency.csv', 'w')
                            self.fconflict = open('output/conflict.csv', 'w')
                            plotter.plot('metrics.metrics.sectorsd', dt=2.5, title='Static Density', xlabel='Time',
                              ylabel='Aircraft count',
                              fig=1)
                            plotter.plot('metrics.metrics.sectorconv', dt=2.5, title='Summed Pairwise Convergence', xlabel='Time',
                              ylabel='Convergence',
                              fig=2)
                            self.effplot = plotter.Plot('metrics.metrics.sectoreff', title='Route Efficiency', plot_type='boxplot', xlabel='Sector',
                              ylabel='Efficiency',
                              fig=3)
                        self.sectors.append(name)
                        self.acinside.append(SectorData())
                        plotter.legend(self.sectors, 1)
                        return (True, 'Added %s to sector list.' % name)
                    return (
                     False, "No area found with name '%s', create it first with one of the shape commands" % name)
            else:
                if name in self.sectors:
                    idx = self.sectors.index(name)
                    self.sectors.pop(idx)
                    return (True, 'Removed %s from sector list.' % name)
                return (False, "No sector registered with name '%s'." % name)


def init_plugin():
    global metrics
    metrics = Metrics()
    config = {'plugin_name':'METRICS_TK',
     'plugin_type':'sim',
     'update_interval':2.5,
     'update':metrics.update,
     'reset':metrics.reset}
    stackfunctions = {'METRICS_TK': [
                    'METRICS_TK ADDSECTOR name',
                    'txt,txt',
                    metrics.stackio,
                    'Print something to the bluesky console based on the flag passed to MYFUN.']}
    return (
     config, stackfunctions)