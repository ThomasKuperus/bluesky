''' Conflict resolution based on the Modified Voltage Potential algorithm. '''
import numpy as np, math
from bluesky import stack
from bluesky.traffic.asas import ConflictResolution
import numpy as np
from bluesky import stack
from bluesky.traffic.asas import ConflictResolution
def init_plugin():
    config = {'plugin_name':'MVPTimeSpace',
     'plugin_type':'sim'}
    return (
     config, {})

class MVPTimeSpace(ConflictResolution):

    def __init__(self):
        super().__init__()
        self.swresohoriz = True
        self.swresospd = False
        self.swresohdg = False
        self.swresovert = False
        mvp_stackfuns = {'RMETHH':[
          'RMETHH [method]',
          '[txt]',
          self.setresometh,
          'Set resolution method to be used horizontally'],
         'RMETHV':[
          'RMETHV [method]',
          '[txt]',
          self.setresometv,
          'Set resolution method to be used vertically']}
        stack.append_commands(mvp_stackfuns)

    def setresometh(self, value=''):
        """ Processes the RMETHH command. Sets swresovert = False"""
        # Acceptable arguments for this command
        options = ["BOTH", "SPD", "HDG", "NONE", "ON", "OFF", "OF"]
        if not value:
            return True, "RMETHH [ON / BOTH / OFF / NONE / SPD / HDG]" + \
                         "\nHorizontal resolution limitation is currently " + ("ON" if self.swresohoriz else "OFF") + \
                         "\nSpeed resolution limitation is currently " + ("ON" if self.swresospd else "OFF") + \
                         "\nHeading resolution limitation is currently " + \
                ("ON" if self.swresohdg else "OFF")
        if value not in options:
            return False, "RMETH Not Understood" + "\nRMETHH [ON / BOTH / OFF / NONE / SPD / HDG]"
        else:
            if value == "ON" or value == "BOTH":
                self.swresohoriz = True
                self.swresospd = True
                self.swresohdg = True
                self.swresovert = False
            elif value == "OFF" or value == "OF" or value == "NONE":
                # Do NOT swtich off self.swresovert if value == OFF
                self.swresohoriz = False
                self.swresospd = False
                self.swresohdg = False
            elif value == "SPD":
                self.swresohoriz = True
                self.swresospd = True
                self.swresohdg = False
                self.swresovert = False
            elif value == "HDG":
                self.swresohoriz = True
                self.swresospd = False
                self.swresohdg = True
                self.swresovert = False


    def setresometv(self, value=''):
        """ Processes the RMETHV command. Sets swresohoriz = False."""
        # Acceptable arguments for this command
        options = ["NONE", "ON", "OFF", "OF", "V/S"]
        if not value:
            return True, "RMETHV [ON / V/S / OFF / NONE]" + \
                         "\nVertical resolution limitation is currently " + \
                ("ON" if self.swresovert else "OFF")
        if value not in options:
            return False, "RMETV Not Understood" + "\nRMETHV [ON / V/S / OFF / NONE]"
        else:
            if value == "ON" or value == "V/S":
                self.swresovert = True
                self.swresohoriz = False
                self.swresospd = False
                self.swresohdg = False
            elif value == "OFF" or value == "OF" or value == "NONE":
                # Do NOT swtich off self.swresohoriz if value == OFF
                self.swresovert = False

    def applyprio(self, dv_mvp, dv1, dv2, vs1, vs2):
        ''' Apply the desired priority setting to the resolution '''

        # Primary Free Flight prio rules (no priority)
        if self.priocode == 'FF1':
            # since cooperative, the vertical resolution component can be halved, and then dv_mvp can be added
            dv_mvp[2] = dv_mvp[2] / 2.0
            dv1 = dv1 - dv_mvp
            dv2 = dv2 + dv_mvp

        # Secondary Free Flight (Cruising aircraft has priority, combined resolutions)
        if self.priocode == 'FF2':
            # since cooperative, the vertical resolution component can be halved, and then dv_mvp can be added
            dv_mvp[2] = dv_mvp[2]/2.0
            # If aircraft 1 is cruising, and aircraft 2 is climbing/descending -> aircraft 2 solves conflict
            if abs(vs1) < 0.1 and abs(vs2) > 0.1:
                dv2 = dv2 + dv_mvp
            # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft 1 solves conflict
            elif abs(vs2) < 0.1 and abs(vs1) > 0.1:
                dv1 = dv1 - dv_mvp
            else:  # both are climbing/descending/cruising -> both aircraft solves the conflict
                dv1 = dv1 - dv_mvp
                dv2 = dv2 + dv_mvp

        # Tertiary Free Flight (Climbing/descending aircraft have priority and crusing solves with horizontal resolutions)
        elif self.priocode == 'FF3':
            # If aircraft 1 is cruising, and aircraft 2 is climbing/descending -> aircraft 1 solves conflict horizontally
            if abs(vs1) < 0.1 and abs(vs2) > 0.1:
                dv_mvp[2] = 0.0
                dv1 = dv1 - dv_mvp
            # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft 2 solves conflict horizontally
            elif abs(vs2) < 0.1 and abs(vs1) > 0.1:
                dv_mvp[2] = 0.0
                dv2 = dv2 + dv_mvp
            else:  # both are climbing/descending/cruising -> both aircraft solves the conflict, combined
                dv_mvp[2] = dv_mvp[2]/2.0
                dv1 = dv1 - dv_mvp
                dv2 = dv2 + dv_mvp

        # Primary Layers (Cruising aircraft has priority and clmibing/descending solves. All conflicts solved horizontally)
        elif self.priocode == 'LAY1':
            dv_mvp[2] = 0.0
            # If aircraft 1 is cruising, and aircraft 2 is climbing/descending -> aircraft 2 solves conflict horizontally
            if abs(vs1) < 0.1 and abs(vs2) > 0.1:
                dv2 = dv2 + dv_mvp
            # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft 1 solves conflict horizontally
            elif abs(vs2) < 0.1 and abs(vs1) > 0.1:
                dv1 = dv1 - dv_mvp
            else:  # both are climbing/descending/cruising -> both aircraft solves the conflict horizontally
                dv1 = dv1 - dv_mvp
                dv2 = dv2 + dv_mvp

        # Secondary Layers (Climbing/descending aircraft has priority and cruising solves. All conflicts solved horizontally)
        elif self.priocode == 'LAY2':
            dv_mvp[2] = 0.0
            # If aircraft 1 is cruising, and aircraft 2 is climbing/descending -> aircraft 1 solves conflict horizontally
            if abs(vs1) < 0.1 and abs(vs2) > 0.1:
                dv1 = dv1 - dv_mvp
            # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft 2 solves conflict horizontally
            elif abs(vs2) < 0.1 and abs(vs1) > 0.1:
                dv2 = dv2 + dv_mvp
            else:  # both are climbing/descending/cruising -> both aircraft solves the conflic horizontally
                dv1 = dv1 - dv_mvp
                dv2 = dv2 + dv_mvp

        return dv1, dv2


    def resolve(self, conf, ownship, intruder):
        """ Resolve all current conflicts """
        dv = np.zeros((ownship.ntraf, 3))
        conf_time = np.zeros((ownship.ntraf, 1))
        conf_count = np.zeros((ownship.ntraf, 1))
        side1 = np.zeros((ownship.ntraf, 1))
        side2 = np.zeros((ownship.ntraf, 1))
        for (ac1, ac2), qdr in zip(conf.confpairs, conf.qdr):
            idx1 = ownship.id.index(ac1)
            conf_count[idx1] += 1
            if ownship.hdg[idx1] < 180:
                if 0 < qdr - ownship.hdg[idx1] <= 180:
                    side1[idx1] += 1
                else:
                    side2[idx1] += 1
            elif -180 < qdr - ownship.hdg[idx1] <= 0:
                side1[idx1] += 1
            else:
                side2[idx1] += 1

        timesolveV = np.ones(ownship.ntraf) * 1000000000.0
        for (ac1, ac2), qdr, dist, tcpa, tLOS in zip(conf.confpairs, conf.qdr, conf.dist, conf.tcpa, conf.tLOS):
            idx1 = ownship.id.index(ac1)
            idx2 = intruder.id.index(ac2)
            if idx1 > -1 and idx2 > -1:
                dv_mvp, tsolV = self.MVP(ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2)
                conf_time[idx1] += self.dtlookahead - tLOS
                if self.swprio:
                    dv[idx1], _ = self.applyprio(dv_mvp, dv[idx1], dv[idx2], ownship.vs[idx1], intruder.vs[idx2])
                else:
                    dv_mvp[2] = 0.5 * dv_mvp[2]
                    hdg_dv = (np.degrees(math.atan2(dv_mvp[1], dv_mvp[0])) + 360) % 360
                    if ownship.hdg[idx1] < 180:
                        if ownship.hdg[idx1] < hdg_dv <= ownship.hdg[idx1] + 180:
                            spaceWeight = 1 / ((side1[idx1] + 1) / ((conf_count[idx1] + 2) / 2))
                        else:
                            spaceWeight = 1 / ((side2[idx1] + 1) / ((conf_count[idx1] + 2) / 2))
                    else:
                        if ownship.hdg[idx1] - 180 < hdg_dv <= ownship.hdg[idx1]:
                            spaceWeight = 1 / ((side1[idx1] + 1) / ((conf_count[idx1] + 2) / 2))
                        else:
                            spaceWeight = 1 / ((side2[idx1] + 1) / ((conf_count[idx1] + 2) / 2))
                    dv[idx1] = dv[idx1] - dv_mvp * (self.dtlookahead - tLOS) * spaceWeight
                if self.noresoac[idx2]:
                    dv[idx1] = dv[idx1] + dv_mvp
                if self.resooffac[idx1]:
                    dv[idx1] = 0.0

        conf_time_avg = np.divide(conf_time, conf_count, out=(np.zeros_like(conf_time)), where=(conf_count != 0))
        dv = np.divide(dv, conf_time_avg, out=(np.zeros_like(dv)), where=(conf_time_avg != 0))
        dv = np.transpose(dv)
        v = np.array([ownship.gseast, ownship.gsnorth, ownship.vs])
        newv = v + dv
        if self.swresohoriz:
            if self.swresospd:
                newtrack = self.swresohdg or ownship.trk
                newgs = np.sqrt(newv[0, :] ** 2 + newv[1, :] ** 2)
                newvs = ownship.vs
            else:
                if self.swresohdg:
                    newtrack = self.swresospd or np.arctan2(newv[0, :], newv[1, :]) * 180 / np.pi % 360
                    newgs = ownship.gs
                    newvs = ownship.vs
                else:
                    newtrack = np.arctan2(newv[0, :], newv[1, :]) * 180 / np.pi % 360
                    newgs = np.sqrt(newv[0, :] ** 2 + newv[1, :] ** 2)
                    newvs = ownship.vs
        else:
            if self.swresovert:
                newtrack = ownship.trk
                newgs = ownship.gs
                newvs = newv[2, :]
            else:
                newtrack = np.arctan2(newv[0, :], newv[1, :]) * 180 / np.pi % 360
                newgs = np.sqrt(newv[0, :] ** 2 + newv[1, :] ** 2)
                newvs = newv[2, :]
        newgscapped = np.maximum(ownship.perf.vmin, np.minimum(ownship.perf.vmax, newgs))
        vscapped = np.maximum(ownship.perf.vsmin, np.minimum(ownship.perf.vsmax, newvs))
        asasalttemp = vscapped * timesolveV + ownship.alt
        signdvs = np.sign(vscapped - ownship.ap.vs * np.sign(ownship.selalt - ownship.alt))
        signalt = np.sign(asasalttemp - ownship.selalt)
        alt = np.where(np.logical_or(signdvs == 0, signdvs == signalt), asasalttemp, ownship.selalt)
        altCondition = np.logical_and(timesolveV < conf.dtlookahead, np.abs(dv[2, :]) > 0.0)
        alt[altCondition] = asasalttemp[altCondition]
        alt = alt * (1 - self.swresohoriz) + ownship.selalt * self.swresohoriz
        return (newtrack, newgscapped, vscapped, alt)

    def MVP(self, ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2):
        """Modified Voltage Potential (MVP) resolution method"""
        # Preliminary calculations-------------------------------------------------

        # Convert qdr from degrees to radians
        qdr = np.radians(qdr)

        # Relative position vector between id1 and id2
        drel = np.array([np.sin(qdr)*dist, \
                        np.cos(qdr)*dist, \
                        intruder.alt[idx2]-ownship.alt[idx1]])

        # Write velocities as vectors and find relative velocity vector
        v1 = np.array([ownship.gseast[idx1], ownship.gsnorth[idx1], ownship.vs[idx1]])
        v2 = np.array([intruder.gseast[idx2], intruder.gsnorth[idx2], intruder.vs[idx2]])
        vrel = np.array(v2-v1)


        # Horizontal resolution----------------------------------------------------

        # Find horizontal distance at the tcpa (min horizontal distance)
        dcpa  = drel + vrel*tcpa
        dabsH = np.sqrt(dcpa[0]*dcpa[0]+dcpa[1]*dcpa[1])

        # Compute horizontal intrusion
        iH = (conf.rpz * self.resofach) - dabsH

        # Exception handlers for head-on conflicts
        # This is done to prevent division by zero in the next step
        if dabsH <= 10.:
            dabsH = 10.
            dcpa[0] = drel[1] / dist * dabsH
            dcpa[1] = -drel[0] / dist * dabsH

        # If intruder is outside the ownship PZ, then apply extra factor
        # to make sure that resolution does not graze IPZ
        if (conf.rpz * self.resofach) < dist and dabsH < dist:
            # Compute the resolution velocity vector in horizontal direction.
            # abs(tcpa) because it bcomes negative during intrusion.
            erratum=np.cos(np.arcsin((conf.rpz * self.resofach)/dist)-np.arcsin(dabsH/dist))
            dv1 = (((conf.rpz * self.resofach)/erratum - dabsH)*dcpa[0])/(abs(tcpa)*dabsH)
            dv2 = (((conf.rpz * self.resofach)/erratum - dabsH)*dcpa[1])/(abs(tcpa)*dabsH)
        else:
            dv1 = (iH * dcpa[0]) / (abs(tcpa) * dabsH)
            dv2 = (iH * dcpa[1]) / (abs(tcpa) * dabsH)

        # Vertical resolution------------------------------------------------------

        # Compute the  vertical intrusion
        # Amount of vertical intrusion dependent on vertical relative velocity
        iV = (conf.hpz * self.resofacv) if abs(vrel[2])>0.0 else (conf.hpz * self.resofacv)-abs(drel[2])

        # Get the time to solve the conflict vertically - tsolveV
        tsolV = abs(drel[2]/vrel[2]) if abs(vrel[2])>0.0 else tLOS

        # If the time to solve the conflict vertically is longer than the look-ahead time,
        # because the the relative vertical speed is very small, then solve the intrusion
        # within tinconf
        if tsolV>conf.dtlookahead:
            tsolV = tLOS
            iV    = (conf.hpz * self.resofacv)

        # Compute the resolution velocity vector in the vertical direction
        # The direction of the vertical resolution is such that the aircraft with
        # higher climb/decent rate reduces their climb/decent rate
        dv3 = np.where(abs(vrel[2])>0.0,  (iV/tsolV)*(-vrel[2]/abs(vrel[2])), (iV/tsolV))

        # It is necessary to cap dv3 to prevent that a vertical conflict
        # is solved in 1 timestep, leading to a vertical separation that is too
        # high (high vs assumed in traf). If vertical dynamics are included to
        # aircraft  model in traffic.py, the below three lines should be deleted.
    #    mindv3 = -400*fpm# ~ 2.016 [m/s]
    #    maxdv3 = 400*fpm
    #    dv3 = np.maximum(mindv3,np.minimum(maxdv3,dv3))


        # Combine resolutions------------------------------------------------------

        # combine the dv components
        dv = np.array([dv1,dv2,dv3])

        return dv, tsolV
