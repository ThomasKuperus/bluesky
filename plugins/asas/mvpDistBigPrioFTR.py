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
    config = {'plugin_name':'MVPDistBigPrioFTR',
     'plugin_type':'sim'}
    return (
     config, {})

class MVPDistBigPrioFTR(MVP):

    def resolve(self, conf, ownship, intruder):
        """ Resolve all current conflicts """
        dv = np.zeros((ownship.ntraf, 3))
        conf_dist = np.zeros((ownship.ntraf, 1))
        conf_count = np.zeros((ownship.ntraf, 1))
        timesolveV = np.ones(ownship.ntraf) * 1000000000.0
        for ((ac1, ac2), qdr, dist, tcpa, tLOS,nolook) in zip(conf.confpairs, conf.qdr, conf.dist, conf.tcpa, conf.tLOS,conf.nolook):
            idx1 = ownship.id.index(ac1)
            idx2 = intruder.id.index(ac2)
            if idx1 > -1 and nolook==False:
                if idx2 > -1:
                    dv_mvp, tsolV = self.MVP(ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2)
                    conf_dist[idx1] += dist
                    conf_count[idx1] += 1
                    if tsolV < timesolveV[idx1]:
                        timesolveV[idx1] = tsolV
                    if self.swprio:
                        dv[idx1], _ = self.applyprio(dv_mvp, dv[idx1], dv[idx2], ownship.vs[idx1], intruder.vs[idx2])
                    else:
                        dv_mvp[2] = 0.5 * dv_mvp[2]
                        dv[idx1] = dv[idx1] - dv_mvp * ( dist)
                    if self.noresoac[idx2]:
                        dv[idx1] = dv[idx1] + dv_mvp
                if self.resooffac[idx1]:
                    dv[idx1] = 0.0

        conf_dist_avg = np.divide(conf_dist, conf_count, out=(np.zeros_like(conf_dist)), where=(conf_count != 0))
        dv = np.divide(dv, conf_dist_avg, out=(np.zeros_like(dv)), where=(conf_dist_avg != 0))
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

    def resumenav(self, conf, ownship, intruder):
        '''
            Decide for each aircraft in the conflict list whether the ASAS
            should be followed or not, based on if the aircraft pairs passed
            their CPA.
        '''
        pass
        # Add new conflicts to resopairs and confpairs_all and new losses to lospairs_all
        self.resopairs.update(conf.confpairs)

        # Conflict pairs to be deleted
        delpairs = set()
        changeactive = dict()

        # Look at all conflicts, also the ones that are solved but CPA is yet to come
        for conflict in self.resopairs:
            idx1, idx2 = bs.traf.id2idx(conflict)
            # If the ownship aircraft is deleted remove its conflict from the list
            if idx1 < 0:
                delpairs.add(conflict)
                continue

            if idx2 >= 0:
                # Distance vector using flat earth approximation
                re = 6371000.
                dist = re * np.array([np.radians(intruder.lon[idx2] - ownship.lon[idx1]) *
                                      np.cos(0.5 * np.radians(intruder.lat[idx2] +
                                                              ownship.lat[idx1])),
                                      np.radians(intruder.lat[idx2] - ownship.lat[idx1])])

                # Relative velocity vector
                vrel = np.array([intruder.gseast[idx2] - ownship.gseast[idx1],
                                 intruder.gsnorth[idx2] - ownship.gsnorth[idx1]])

                # Check if conflict is past CPA
                past_cpa = np.dot(dist, vrel) > 0.0

                # Free to Resume Method
                des_aphdg_1 = bs.traf.ap.trk[idx1]
                des_aphdg_2 = bs.traf.ap.trk[idx2]

                des_apspd_1 = bs.traf.ap.tas[idx1]
                des_apspd_2 = bs.traf.ap.tas[idx2]

                des_gsnorth_1 = des_apspd_1 * np.cos(np.radians(des_aphdg_1))
                des_gseast_1 = des_apspd_1 * np.sin(np.radians(des_aphdg_1))

                des_gsnorth_2 = des_apspd_2 * np.cos(np.radians(des_aphdg_2))
                des_gseast_2 = des_apspd_2 * np.sin(np.radians(des_aphdg_2))

                des_vrel = np.array([des_gseast_2 - des_gseast_1,
                                     des_gsnorth_2 - des_gsnorth_1])

                des_tcpa_1 = np.maximum(-(des_vrel[0] * dist[0] + des_vrel[1] * dist[1]) / (des_vrel[0]*des_vrel[0] + des_vrel[1]*des_vrel[1]),0.0)
                des_dcpa_1 = dist + des_vrel*des_tcpa_1
                free = np.linalg.norm(des_dcpa_1) > bs.traf.cd.rpz
                # End of free to resume method

                # hor_los:
                # Aircraft should continue to resolve until there is no horizontal
                # LOS. This is particularly relevant when vertical resolutions
                # are used.
                hdist = np.linalg.norm(dist)
                hor_los = hdist < conf.rpz

                # Bouncing conflicts:
                # If two aircraft are getting in and out of conflict continously,
                # then they it is a bouncing conflict. ASAS should stay active until
                # the bouncing stops.
                is_bouncing = abs(
                    ownship.trk[idx1] - intruder.trk[idx2]) < 30.0 and hdist < conf.rpz * self.resofach

            # Start recovery for ownship if intruder is deleted, or if past CPA
            # and not in horizontal LOS or a bouncing conflict
            if idx2 >= 0 and (not free or hor_los or is_bouncing):
                # Enable ASAS for this aircraft
                changeactive[idx1] = True
            else:
                # Switch ASAS off for ownship if there are no other conflicts
                # that this aircraft is involved in.
                changeactive[idx1] = changeactive.get(idx1, False)
                # If conflict is solved, remove it from the resopairs list
                delpairs.add(conflict)

        for idx, active in changeactive.items():
            # Loop a second time: this is to avoid that ASAS resolution is
            # turned off for an aircraft that is involved simultaneously in
            # multiple conflicts, where the first, but not all conflicts are
            # resolved.
            self.active[idx] = active
            if not active:
                # Waypoint recovery after conflict: Find the next active waypoint
                # and send the aircraft to that waypoint.
                iwpid = bs.traf.ap.route[idx].findact(idx)
                if iwpid != -1:  # To avoid problems if there are no waypoints
                    bs.traf.ap.route[idx].direct(
                        idx, bs.traf.ap.route[idx].wpname[iwpid])

        # Remove pairs from the list that are past CPA or have deleted aircraft
        self.resopairs -= delpairs
