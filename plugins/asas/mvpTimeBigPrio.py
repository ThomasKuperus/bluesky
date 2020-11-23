''' Conflict resolution based on the Modified Voltage Potential algorithm. '''
import numpy as np
from bluesky import stack
from bluesky.traffic.asas import ConflictResolution
import numpy as np
from bluesky import stack
from bluesky.traffic.asas import ConflictResolution
from bluesky.traffic.asas import MVP
import bluesky as bs

def init_plugin():
    config = {'plugin_name':'MVPTimeBigPrio',
     'plugin_type':'sim'}
    return (
     config, {})

class MVPTimeBigPrio(MVP):

    def resolve(self, conf, ownship, intruder):
        """ Resolve all current conflicts """
        dv = np.zeros((ownship.ntraf, 3))
        tLOS_all = np.zeros((ownship.ntraf, 1))
        conf_count = np.zeros((ownship.ntraf, 1))
        timesolveV = np.ones(ownship.ntraf) * 1000000000.0
        for ((ac1, ac2), qdr, dist, tcpa, tLOS,nolook) in zip(conf.confpairs, conf.qdr, conf.dist, conf.tcpa, conf.tLOS,conf.nolook):
            idx1 = ownship.id.index(ac1)
            idx2 = intruder.id.index(ac2)
            if idx1 > -1 and nolook==False:
                if idx2 > -1:
                    dv_mvp, tsolV = self.MVP(ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2)
                    tLOS_all[idx1] += tLOS
                    conf_count[idx1] += 1
                    if tsolV < timesolveV[idx1]:
                        timesolveV[idx1] = tsolV
                    if self.swprio:
                        dv[idx1], _ = self.applyprio(dv_mvp, dv[idx1], dv[idx2], ownship.vs[idx1], intruder.vs[idx2])
                    else:
                        dv_mvp[2] = 0.5 * dv_mvp[2]
                        dv[idx1] = dv[idx1] - dv_mvp * (tLOS)
                    if self.noresoac[idx2]:
                        dv[idx1] = dv[idx1] + dv_mvp
                if self.resooffac[idx1]:
                    dv[idx1] = 0.0

        tLOS_all_avg = np.divide(tLOS_all, conf_count, out=(np.zeros_like(tLOS_all)), where=(conf_count != 0))
        dv = np.divide(dv, tLOS_all_avg, out=(np.zeros_like(dv)), where=(tLOS_all_avg != 0))
        dv = np.transpose(dv)
        v = np.array([ownship.gseast, ownship.gsnorth, ownship.vs])
        newv = v + dv
        if self.swresohoriz:  # horizontal resolutions
            if self.swresospd and not self.swresohdg:  # SPD only
                newtrack = ownship.trk
                newgs = np.sqrt(newv[0, :] ** 2 + newv[1, :] ** 2)
                newvs = ownship.vs
            elif self.swresohdg and not self.swresospd:  # HDG only
                newtrack = (np.arctan2(newv[0, :], newv[1, :]) * 180 / np.pi) % 360
                newgs = ownship.gs
                newvs = ownship.vs
            else:  # SPD + HDG
                newtrack = (np.arctan2(newv[0, :], newv[1, :]) * 180 / np.pi) % 360
                newgs = np.sqrt(newv[0, :] ** 2 + newv[1, :] ** 2)
                newvs = ownship.vs
        elif self.swresovert:  # vertical resolutions
            newtrack = ownship.trk
            newgs = ownship.gs
            newvs = newv[2, :]
        else:  # horizontal + vertical
            newtrack = (np.arctan2(newv[0, :], newv[1, :]) * 180 / np.pi) % 360
            newgs = np.sqrt(newv[0, :] ** 2 + newv[1, :] ** 2)
            newvs = newv[2, :]

        # Determine ASAS module commands for all aircraft--------------------------

        # Cap the velocity
        newgscapped = np.maximum(ownship.perf.currentlimits()[0], np.minimum(ownship.perf.currentlimits()[1], newgs))
        # print(ownship.perf.currentlimits()[0], vtas2mach(ownship.perf.currentlimits()[0],ownship.alt))
        # print(ownship.perf.currentlimits()[1],vtas2mach(ownship.perf.currentlimits()[1],ownship.alt))
        # Cap the vertical speed
        vscapped = np.maximum(ownship.perf.vsmin, np.minimum(ownship.perf.vsmax, newvs))

        # Calculate if Autopilot selected altitude should be followed. This avoids ASAS from
        # climbing or descending longer than it needs to if the autopilot leveloff
        # altitude also resolves the conflict. Because asasalttemp is calculated using
        # the time to resolve, it may result in climbing or descending more than the selected
        # altitude.
        asasalttemp = vscapped * timesolveV + ownship.alt
        signdvs = np.sign(vscapped - ownship.ap.vs * np.sign(ownship.selalt - ownship.alt))
        signalt = np.sign(asasalttemp - ownship.selalt)
        alt = np.where(np.logical_or(signdvs == 0, signdvs == signalt), asasalttemp, ownship.selalt)

        # To compute asas alt, timesolveV is used. timesolveV is a really big value (1e9)
        # when there is no conflict. Therefore asas alt is only updated when its
        # value is less than the look-ahead time, because for those aircraft are in conflict
        altCondition = np.logical_and(timesolveV < conf.dtlookahead, np.abs(dv[2, :]) > 0.0)
        alt[altCondition] = asasalttemp[altCondition]

        # If resolutions are limited in the horizontal direction, then asasalt should
        # be equal to auto pilot alt (aalt). This is to prevent a new asasalt being computed
        # using the auto pilot vertical speed (ownship.avs) using the code in line 106 (asasalttemp) when only
        # horizontal resolutions are allowed.
        alt = alt * (1 - self.swresohoriz) + ownship.selalt * self.swresohoriz
        # print('final', newtrack, newgscapped)
        return (newtrack, newgscapped, vscapped, alt,dv)
