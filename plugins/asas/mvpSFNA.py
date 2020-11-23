''' Conflict resolution based on the Modified Voltage Potential algorithm. '''
import numpy as np
from bluesky import stack
from bluesky.traffic.asas import ConflictResolution
from bluesky.tools.aero import vcas2tas, vcas2mach, vtas2mach
from bluesky.traffic.asas import MVP
import bluesky as bs

def init_plugin():
    #Smallest dv first
    config = {'plugin_name':'MVPSFNA',
     'plugin_type':'sim'}
    return (
     config, {})

class MVPSFNA(MVP):

    def resolve(self, conf, ownship, intruder):
        ''' Resolve all current conflicts '''
        # Initialize an array to store the resolution velocity vector for all A/C
        dv = np.zeros((ownship.ntraf, 3))

        # Initialize an array to store time needed to resolve vertically
        timesolveV = np.ones(ownship.ntraf) * 1e9

        sol_mat_depth = 1  # depth of solution matrix, must equal the maximum number of conflicts, this number gets bigger in itterations
        nac = ownship.ntraf #Number of aircraft
        conf_count = np.zeros(ownship.ntraf) #Number of conflicts per ownship
        dv_mvp_all = np.zeros((nac, nac,3)) #Collect all dv

        # Call MVP function to resolve conflicts-----------------------------------
        for ((ac1, ac2), qdr, dist, tcpa, tLOS,nolook) in zip(conf.confpairs, conf.qdr, conf.dist, conf.tcpa, conf.tLOS,conf.nolook):
            idx1 = ownship.id.index(ac1)
            idx2 = intruder.id.index(ac2)

            # If A/C indexes are found, then apply MVP on this conflict pair
            # Because ADSB is ON, this is done for each aircraft separately
            if idx1 >-1 and idx2 > -1 and nolook==False:
                #conf_count[idx1] = conf_count[idx1] + 1
                #if conf_count[idx1]>sol_mat_depth:
                #    sol_mat_depth = sol_mat_depth+1
                #    dv_mvp_all = np.dstack([dv_mvp_all, np.zeros((nac, 1, 3))])

                # Collect all resolution vectors
                dv_mvp_all[idx1,idx2,:], tsolV = self.MVP(ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2)
                if tsolV < timesolveV[idx1]:
                    timesolveV[idx1] = tsolV

        #divide them into sides (port and starboard)
        dv_mvp_all[:,:,2] = 0 #Dont take vertical resolution in consideration
        v_all = np.array([ownship.gseast, ownship.gsnorth, ownship.vs]).T
        v3d = np.reshape(np.repeat(v_all,nac,axis=0),[nac,nac,3])
        v3d_abs = np.linalg.norm(v3d, axis=2)
        dv_abs = np.linalg.norm(dv_mvp_all, axis=2)

        sides = np.arcsin((np.cross(v3d, dv_mvp_all)[:,:,-1] / (v3d_abs* dv_abs)))
        s1 = np.where(sides < 0, True, False)
        s2 = ~s1

        dv_abs1 = np.sum(dv_abs * s1, axis=1)
        dv_abs2 = np.sum(dv_abs * s2, axis=1)

        #Add a penalty to the side of a 0 score.
        dv_abs1 = np.where(dv_abs1==0,10000,dv_abs1)
        dv_abs2 = np.where(dv_abs2 == 0, 10000, dv_abs2)

        #print('dv',dv_mvp_all)
        for i in range(len(dv_abs)):
            if dv_abs1[i] < dv_abs2[i]:
                dv_mvp_all[i] = ((dv_mvp_all[i].T * s1[i]).T)#/np.sum(s1[i])
            elif dv_abs1[i] > dv_abs2[i]:
                dv_mvp_all[i] = ((dv_mvp_all[i].T * s2[i]).T)#/np.sum(s2[i])
            #else:
            #    dv_mvp_all[i] = dv_mvp_all[i] /(np.sum(s1[i])+np.sum(s2[i]))

        #print(dv_mvp_all)
        dv = np.sum(dv_mvp_all, axis=1)
        #print('dv', dv)
        # Determine new speed and limit resolution direction for all aicraft-------

        # Resolution vector for all aircraft, cartesian coordinates
        dv = np.transpose(dv)
        # The old speed vector, cartesian coordinates
        v = np.array([ownship.gseast, ownship.gsnorth, ownship.vs])
        #print('v',v)
        #print('vabs',np.sqrt(v[0]*v[0]+v[1]*v[1]))
        #print('hdg', ownship.hdg)
        # The new speed vector, cartesian coordinates
        newv = v - dv
        # Limit resolution direction if required-----------------------------------
        #print('newv',newv)
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
        #print(ownship.perf.currentlimits()[0], vtas2mach(ownship.perf.currentlimits()[0],ownship.alt))
        #print(ownship.perf.currentlimits()[1],vtas2mach(ownship.perf.currentlimits()[1],ownship.alt))
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
        #print('final', newtrack, newgscapped)
        return newtrack, newgscapped, vscapped, alt,dv