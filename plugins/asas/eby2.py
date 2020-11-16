# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 18:24:37 2020

@author: thoma
"""

import numpy as np
from bluesky.tools.aero import vtas2eas
from bluesky.traffic.asas import ConflictResolution
from bluesky import stack

def init_plugin():
    config = {'plugin_name':'EBY2',
     'plugin_type':'sim'}
    return (
     config, {})


class Eby2(ConflictResolution):

    def resolve(self, conf, ownship, intruder):
        """ Resolve all current conflicts """
        dv = np.zeros((ownship.ntraf, 3))
        for (ac1, ac2), qdr, dist, tcpa, tLOS in zip(conf.confpairs, conf.qdr, conf.dist, conf.tcpa, conf.tLOS):
            idx1 = ownship.id.index(ac1)
            idx2 = intruder.id.index(ac2)
            if idx1 > -1 and idx2 > -1:
                dv_eby = self.Eby_straight(ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2)
                dv[idx1] -= dv_eby
                dv[idx2] += dv_eby

        dv = np.transpose(dv)
        trkrad = np.radians(ownship.trk)
        v = np.array([np.sin(trkrad) * ownship.tas,
         np.cos(trkrad) * ownship.tas,
         ownship.vs])
        W = 2
        stack.stack('ECHO yes')
        if ownship.trk[0] > 180:
            newv = dv * W + v
        else:
            newv = dv / W + v
        newtrack = np.arctan2(newv[0, :], newv[1, :]) * 180 / np.pi % 360
        newgs = np.sqrt(newv[0, :] ** 2 + newv[1, :] ** 2)
        neweas = vtas2eas(newgs, ownship.alt)
        neweascapped = np.maximum(ownship.perf.vmin, np.minimum(ownship.perf.vmax, neweas))
        return (
         newtrack, neweascapped, newv[2, :], np.sign(newv[2, :]) * 100000.0)

    def Eby_straight(self, ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2):
        """
            Resolution: Eby method assuming aircraft move straight forward,
            solving algebraically, only horizontally.
        """
        qdr = np.radians(qdr)
        d = np.array([np.sin(qdr) * dist,
         np.cos(qdr) * dist,
         intruder.alt[idx2] - ownship.alt[idx1]])
        t1 = np.radians(ownship.trk[idx1])
        t2 = np.radians(intruder.trk[idx2])
        v1 = np.array([np.sin(t1) * ownship.tas[idx1], np.cos(t1) * ownship.tas[idx1], ownship.vs[idx1]])
        v2 = np.array([np.sin(t2) * intruder.tas[idx2], np.cos(t2) * intruder.tas[idx2], intruder.vs[idx2]])
        v = np.array(v2 - v1)
        R2 = (conf.rpz * self.resofach) ** 2
        d2 = np.dot(d, d)
        v2 = np.dot(v, v)
        dv = np.dot(d, v)
        a = R2 * v2 - dv ** 2
        b = 2 * dv * (R2 - d2)
        c = R2 * d2 - d2 ** 2
        discrim = b ** 2 - 4 * a * c
        if discrim < 0:
            discrim = 0
        time1 = (-b + np.sqrt(discrim)) / (2 * a)
        time2 = (-b - np.sqrt(discrim)) / (2 * a)
        tstar = min(abs(time1), abs(time2))
        drelstar = d + v * tstar
        dstarabs = np.linalg.norm(drelstar)
        exactcourse = 10
        dif = exactcourse - dstarabs
        if dif > 0:
            vperp = np.array([-v[1], v[0], 0])
            drelstar += dif * vperp / np.linalg.norm(vperp)
            dstarabs = np.linalg.norm(drelstar)
        i = conf.rpz * self.resofach - dstarabs
        dv = i * drelstar / (dstarabs * tstar)
        return dv