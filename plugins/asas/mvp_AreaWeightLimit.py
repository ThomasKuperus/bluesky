''' Conflict resolution based on the Modified Voltage Potential algorithm. '''
import numpy as np
from bluesky import stack
from bluesky.traffic.asas import ConflictResolution
from bluesky.tools.aero import vcas2tas, vcas2mach, vtas2mach

def init_plugin():
    config = {'plugin_name':'MVPAreaWLimit',
     'plugin_type':'sim'}
    return (
     config, {})

class MVPAreaWLimit(ConflictResolution):
    def __init__(self):
        super().__init__()
        # [-] switch to limit resolution to the horizontal direction
        self.swresohoriz = True
        # [-] switch to use only speed resolutions (works with swresohoriz = True)
        self.swresospd = False
        # [-] switch to use only heading resolutions (works with swresohoriz = True)
        self.swresohdg = False
        # [-] switch to limit resolution to the vertical direction
        self.swresovert = False

        mvp_stackfuns = {
            "RMETHH": [
                "RMETHH [method]",
                "[txt]",
                self.setresometh,
                "Set resolution method to be used horizontally",
            ],
            "RMETHV": [
                "RMETHV [method]",
                "[txt]",
                self.setresometv,
                "Set resolution method to be used vertically",
            ]}
        stack.append_commands(mvp_stackfuns)

    def setprio(self, flag=None, priocode=''):
        '''Set the prio switch and the type of prio '''
        if flag is None:
            return True, "PRIORULES [ON/OFF] [PRIOCODE]" + \
                            "\nAvailable priority codes: " + \
                            "\n     FF1:  Free Flight Primary (No Prio) " + \
                            "\n     FF2:  Free Flight Secondary (Cruising has priority)" + \
                            "\n     FF3:  Free Flight Tertiary (Climbing/descending has priority)" + \
                            "\n     LAY1: Layers Primary (Cruising has priority + horizontal resolutions)" + \
                            "\n     LAY2: Layers Secondary (Climbing/descending has priority + horizontal resolutions)" + \
                            "\nPriority is currently " + ("ON" if self.swprio else "OFF") + \
                            "\nPriority code is currently: " + \
                str(self.priocode)
        options = ["FF1", "FF2", "FF3", "LAY1", "LAY2"]
        if priocode not in options:
            return False, "Priority code Not Understood. Available Options: " + str(options)
        return super().setprio(flag, priocode)

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
        ''' Resolve all current conflicts '''
        # Initialize an array to store the resolution velocity vector for all A/C

        sol_mat_depth = 1  # depth of solution matrix, must equal the maximum number of conflicts, this number gets bigger in itterations
        conf_count = np.zeros((ownship.ntraf, sol_mat_depth))

        nac = ownship.ntraf
        dv_mvp_all = np.zeros((nac, 3,sol_mat_depth))
        dv_mvp_opp_all = np.zeros((nac, 3,sol_mat_depth))
        v1_all = np.zeros((nac, 3,sol_mat_depth))
        v2_all = np.zeros((nac, 3,sol_mat_depth))
        vrel_all = np.zeros((nac, 3,sol_mat_depth))
        weights = np.zeros((nac, 1,sol_mat_depth))
        # store angle solutions
        dv_sum = np.zeros((ownship.ntraf,3))
        #idx1 = np.zeros(nac)
        #idx2 = np.zeros(nac)
        # Initialize an array to store time needed to resolve vertically
        timesolveV = np.ones(ownship.ntraf) * 1e9
        weights = np.ones([nac,1])
        nconf= np.zeros([nac, 1])
        #ownshipsID, intrudersID = zip(*conf.confpairs)
        ownshipsIdx = np.zeros((len(conf.confpairs),3))

        # Call MVP function to resolve conflicts-----------------------------------
        i=0
        for ((ac1, ac2), qdr, dist, tcpa, tLOS) in zip(conf.confpairs, conf.qdr, conf.dist, conf.tcpa, conf.tLOS):
            idx1 = ownship.id.index(ac1)
            idx2 = intruder.id.index(ac2)
            ownshipsIdx[i]=idx1
            i=i+1
            # If A/C indexes are found, then apply MVP on this conflict pair
            # Because ADSB is ON, this is done for each aircraft separately


            if idx1 >-1 and idx2 > -1:
                conf_count[idx1] = conf_count[idx1] + 1
                if conf_count[idx1]>sol_mat_depth:
                    sol_mat_depth = sol_mat_depth+1
                    dv_mvp_all = np.dstack([dv_mvp_all, np.zeros((nac, 3, 1))])
                    dv_mvp_opp_all = np.dstack([dv_mvp_opp_all, np.zeros((nac, 3, 1))])
                    v1_all = np.dstack([v1_all, np.zeros((nac, 3, 1))])
                    v2_all = np.dstack([v2_all, np.zeros((nac, 3, 1))])
                    vrel_all = np.dstack([vrel_all, np.zeros((nac, 3, 1))])
                    weights = np.hstack([weights, np.ones([nac,1])])

                dv_mvp_all[idx1,:,int(conf_count[idx1]-1)], tsolV, dv_mvp_opp_all[idx1,:,int(conf_count[idx1]-1)],v1_all[idx1,:,int(conf_count[idx1]-1)],v2_all[idx1,:,int(conf_count[idx1]-1)],vrel_all[idx1,:,int(conf_count[idx1]-1)] = self.MVP(ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2)
                if tsolV < timesolveV[idx1]:
                    timesolveV[idx1] = tsolV

        dv_mvp_all = -dv_mvp_all
        dv_mvp_opp_all= -dv_mvp_opp_all

        for ac1 in range(len(dv_mvp_all)):
            dv_mvp = dv_mvp_all[ac1,:,:]
            dv_mvp_opp = dv_mvp_opp_all[ac1, :, :]
            v1 = v1_all[ac1, :, :]
            v2 = v2_all[ac1, :, :]
            vrel = -vrel_all[ac1, :, :] #This calculation needs v1-v2, not v2-v1




            #Only horizontal
            dv_mvp=dv_mvp[:-1]
            nconfac1 = np.count_nonzero(np.count_nonzero(dv_mvp, axis=0))
            if nconfac1>0:
                nconf[ac1] = 1/nconfac1

                #Only select 2D resolution and delete the empty columns
                dv_mvp=dv_mvp[:,:nconfac1]
                dv_mvp_opp = dv_mvp_opp[:-1,:nconfac1]
                v1 = v1[:-1,:nconfac1]
                v2 = v2[:-1,:nconfac1]
                vrel = vrel[:-1,:nconfac1]



                vrel_mag = np.sqrt(vrel[0] * vrel[0] + vrel[1] * vrel[1])


                #print('vrel',vrel)
                #print('vrel_mag', vrel_mag)
                dv_mag = np.sqrt(dv_mvp[0] * dv_mvp[0] + dv_mvp[1] * dv_mvp[1])
                dv_mag_opp = np.sqrt(dv_mvp_opp[0] * dv_mvp_opp[0] + dv_mvp_opp[1] * dv_mvp_opp[1])
                v2_mag =  np.sqrt(v2[0] * v2[0] + v2[1] * v2[1])




                # only for horizontal solutions
                vgraze = dv_mvp + vrel
                vgraze_mag = np.sqrt(vgraze[0] * vgraze[0] + vgraze[1] * vgraze[1])
                #print(vgraze,dv_mvp,vrel)
                vgraze_opp = dv_mvp_opp + vrel
                vgraze_mag_opp = np.sqrt(vgraze_opp[0] * vgraze_opp[0] + vgraze_opp[1] * vgraze_opp[1])


                #Calculate the angles of the SSD
                #print('cross',np.cross(vrel.T, vgraze.T))
                #print('vrel',vrel)
                #print('vgraze',vgraze)
                #print(vrel_mag, vgraze_mag)
                #print(vrel_mag * vgraze_mag)
                #print('vrel',(vrel.T, vgraze.T))
                angles_bounds =np.zeros([np.shape(dv_mvp)[1], 2]) # np.zeros([nconfac1, 2])#
                angles_bounds[:, 0] = np.arcsin(np.cross(vrel.T, vgraze.T) / (vrel_mag * vgraze_mag))
                angles_bounds[:, 1] = np.arcsin(np.cross(vrel.T, vgraze_opp.T) / (vrel_mag * vgraze_mag_opp))
                angles_bounds.sort(axis=1)
                angles_bounds = np.round(angles_bounds,4)
                #print('angles_bounds', angles_bounds)
                #print(np.shape(dv_mvp))


                for ac2 in range(nconfac1):
                    #print('v2',v2)
                    #print('vsol',vsol)


                    if np.shape(dv_mvp)[1] == 1:
                        vsol = vrel + dv_mvp
                        vsol_mag = np.sqrt(vsol[0] * vsol[0] + vsol[1] * vsol[1])
                    else:
                        vsol = (vrel.T + dv_mvp[:,ac2]).T
                        vsol_mag = np.sqrt(vsol[0] * vsol[0] + vsol[1] * vsol[1])

                    print('vrel',vrel)
                    print('dv',dv_mvp)
                    print('vsol',vsol)

                    angles_sol = np.arcsin(np.cross(vrel.T, vsol.T) / (vrel_mag * vsol_mag))
                    angles_sol = np.round(angles_sol, 4)
                    w_sol = np.where((angles_sol > angles_bounds[:, 0]) & (angles_sol < angles_bounds[:, 1]), 0, 1)
                    print(ac1,ac2)
                    print('w_sol',w_sol)
                    print('bounds', angles_bounds)
                    print('asol',angles_sol)
                    weights[ac1,ac2] = weights[ac1,ac2] + sum(w_sol)
                    weights[ac1,:nconfac1] = weights[ac1,:len(w_sol)]-w_sol

        #print(weights)
        weights = weights*nconf
        weights3D = np.moveaxis(np.dstack([weights,weights,weights]),1,2)
        dv_mvp_weighted = dv_mvp_all*weights3D
        dv_sum = np.sum(dv_mvp_weighted,axis=2)

        print('normal',weights)
        print('dv all', dv_mvp_all)
        #print('weighted',dv_mvp_weighted)
        #print('sum', dv_sum)

        dv = dv_sum
        # Resolution vector for all aircraft, cartesian coordinates
        dv = np.transpose(dv)
        # The old speed vector, cartesian coordinates
        v = np.array([ownship.gseast, ownship.gsnorth, ownship.vs])
        # The new speed vector, cartesian coordinates, only horzontal changes
        ownshipsIdx = np.unique(ownshipsIdx).astype(int)
        newv = v + dv

        #print(ownshipsIdx)
        #print(dv)
        #print(newv)
        # Limit resolution direction if required-----------------------------------
        # Compute new speed vector in polar coordinates based on desired resolution
        if self.swresohoriz: # horizontal resolutions
            if self.swresospd and not self.swresohdg: # SPD only
                newtrack = ownship.trk
                newgs    = np.sqrt(newv[0,:]**2 + newv[1,:]**2)
                newvs    = ownship.vs
            elif self.swresohdg and not self.swresospd: # HDG only
                newtrack = (np.arctan2(newv[0,:],newv[1,:])*180/np.pi) % 360
                newgs    = ownship.gs
                newvs    = ownship.vs
            else: # SPD + HDG
                newtrack = (np.arctan2(newv[0,:],newv[1,:])*180/np.pi) %360
                newgs    = np.sqrt(newv[0,:]**2 + newv[1,:]**2)
                newvs    = ownship.vs
        elif self.swresovert: # vertical resolutions
            newtrack = ownship.trk
            newgs    = ownship.gs
            newvs    = newv[2,:]
        else: # horizontal + vertical
            newtrack = (np.arctan2(newv[0,:],newv[1,:])*180/np.pi) %360
            newgs    = np.sqrt(newv[0,:]**2 + newv[1,:]**2)
            newvs    = newv[2,:]

        # Determine ASAS module commands for all aircraft--------------------------

        # Cap the velocity
        newgscapped = np.maximum(ownship.perf.currentlimits()[0],np.minimum(ownship.perf.currentlimits()[1],newgs))
        # Cap the vertical speed
        vscapped = np.maximum(ownship.perf.vsmin,np.minimum(ownship.perf.vsmax,newvs))

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
        altCondition = np.logical_and(timesolveV<conf.dtlookahead, np.abs(dv[2,:])>0.0)
        alt[altCondition] = asasalttemp[altCondition]

        # If resolutions are limited in the horizontal direction, then asasalt should
        # be equal to auto pilot alt (aalt). This is to prevent a new asasalt being computed
        # using the auto pilot vertical speed (ownship.avs) using the code in line 106 (asasalttemp) when only
        # horizontal resolutions are allowed.
        alt = alt * (1 - self.swresohoriz) + ownship.selalt * self.swresohoriz
        return newtrack, newgscapped, vscapped, alt

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

        # Opposing Horizontal resolution----------------------------------------------------
        # Compute horizontal intrusion
        iHopp = (conf.rpz * self.resofach) + dabsH

        # If intruder is outside the ownship PZ, then apply extra factor
        # to make sure that resolution does not graze IPZ
        if (conf.rpz * self.resofach) < dist and dabsH < dist:
            # Compute the resolution velocity vector in horizontal direction.
            # abs(tcpa) because it bcomes negative during intrusion.
            erratum=np.cos(np.arcsin((conf.rpz * self.resofach)/dist)-np.arcsin(dabsH/dist))
            dv1opp = -(((conf.rpz * self.resofach)/erratum + dabsH)*dcpa[0])/(abs(tcpa)*dabsH)
            dv2opp = -(((conf.rpz * self.resofach)/erratum + dabsH)*dcpa[1])/(abs(tcpa)*dabsH)
        else:
            dv1opp = -(iHopp * dcpa[0]) / (abs(tcpa) * dabsH)
            dv2opp = -(iHopp * dcpa[1]) / (abs(tcpa) * dabsH)


        # Vertical resolution------------------------------------------------------

        # Compute the  vertical intrusion
        # Amount of vertical intrusion dependent on vertical relative velocity
        iV = (conf.hpz * self.resofacv) if abs(vrel[2])>0.0 else (conf.hpz * self.resofacv)-abs(drel[2])

        # Get the time to solve the conflict vertically - tsolveV
        tsolV = abs(drel[2]/vrel[2]) if abs(vrel[2])>0.0 else tLOS

        # If the time to solve the conflict vertically is longer than the look-ahead time,
        # because the the relative vertical speed is very small, then solve the intrusion
        # within tinac
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
        dvoppH = np.array([dv1opp, dv2opp, dv3])
        return dv, tsolV, dvoppH,v1,v2,vrel
