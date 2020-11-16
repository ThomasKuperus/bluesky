import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Set variables
update_interval = 2.5 #seconds
rpz = 5*1852. #Radius protected zone in miles
hpz = 1000*0.3048
min_sep_time = 30

#Read scenario cration file
scn = open("C:/Users/thoma/Documents/GitHub/bluesky/scenario/Heading-Normal-UA-Inst288-Rep1.scn", "r")
scn_line = scn.readlines()
create_list = []
for line in scn_line:
    if 'CRE' in line:
        create_list = create_list + [line.replace('>', ',')]

create = pd.DataFrame([sub.split(",") for sub in create_list])
create['spawns'] = pd.to_timedelta(create[0]).dt.total_seconds()
create = create.set_index(2)
create_time = create['spawns']

folder = "C:/Users/thoma/Documents/Master/Afstudeer_Opdracht/Python/Test_results/Heading-Normal-UA-Inst288-Rep1-1hr/"
#Read simulation data
tlos_conf = pd.read_csv(folder+"tlos_conf_all.csv",index_col=0)
qdr_conf = pd.read_csv(folder+"qdr_conf_all.csv",index_col=0)
dist_conf = pd.read_csv(folder+"dist_conf_all.csv",index_col=0)
tcpa_conf = pd.read_csv(folder+"tcpa_conf_all.csv",index_col=0)
dcpa_conf = pd.read_csv(folder+"dcpa_conf_all.csv",index_col=0)
dalt_conf = pd.read_csv(folder+"dalt_conf_all.csv",index_col=0)

tlos_los = pd.read_csv(folder+"tlos_los_all.csv",index_col=0)
qdr_los = pd.read_csv(folder+"qdr_los_all.csv",index_col=0) #angle between ownship and intruder
dist_los = pd.read_csv(folder+"dist_los_all.csv",index_col=0) #distance between ownship and intruder
tcpa_los = pd.read_csv(folder+"tcpa_los_all.csv",index_col=0) #time to cpa
dcpa_los = pd.read_csv(folder+"dcpa_los_all.csv",index_col=0) #distance to cpa
dalt_los = pd.read_csv(folder+"dalt_los_all.csv",index_col=0) #vertical distance between ownship and intruder

tlos_los = tlos_los.transpose()
qdr_los  =  qdr_los.transpose()
dist_los = dist_los.transpose()
tcpa_los = tcpa_los.transpose()
dcpa_los = dcpa_los.transpose()
dalt_los = abs(dalt_los.transpose())

tlos_conf = tlos_conf.transpose()
qdr_conf  =  qdr_conf.transpose()
dist_conf = dist_conf.transpose()
tcpa_conf = tcpa_conf.transpose()
dcpa_conf = dcpa_conf.transpose()
dalt_conf = abs(dalt_conf.transpose())

multi_ac_conf = tlos_conf.notna()
multi_ac_conf  = pd.DataFrame.multiply(multi_ac_conf,multi_ac_conf.index.str[:6], axis=0)
multi_ac_conf2 = multi_ac_conf.apply(pd.value_counts)
multi_ac_conf2 = multi_ac_conf2.drop(index='')
multi_ac_conf2 = multi_ac_conf2.transpose()
#multi_ac_conf2_max = pd.DataFrame()
#multi_ac_conf2_max['max_conflicts'] = multi_ac_conf2.max(axis=1)
#multi_ac_conf2_max = multi_ac_conf2_max[multi_ac_conf2_max['max_conflicts']>1]

#%%Unique pairs
los_pairs = []
for i in list(tlos_los.index)[0:]:
    if i[:6]<i[6:]:
        los_pairs.append(i)

los_data = pd.DataFrame(index=los_pairs) #Include: Time in conflict, angle at start, angle at end, max intrusion, etc.

tlos_los = pd.merge(tlos_los, los_data, how= 'inner',left_index=True, right_index=True,)
qdr_los  = pd.merge(qdr_los , los_data, how= 'inner',left_index=True, right_index=True,)
dist_los = pd.merge(dist_los, los_data, how= 'inner',left_index=True, right_index=True,)
tcpa_los = pd.merge(tcpa_los, los_data, how= 'inner',left_index=True, right_index=True,)
dcpa_los = pd.merge(dcpa_los, los_data, how= 'inner',left_index=True, right_index=True,)
dalt_los = pd.merge(dalt_los, los_data, how= 'inner',left_index=True, right_index=True,)


conf_pairs = []
for i in list(tlos_conf.index)[0:]:
    if i[:6] < i[6:]:
        conf_pairs.append(i)

conf_data = pd.DataFrame(index=conf_pairs)  # Include: Time in conflict, angle at start, angle

tlos_conf = pd.merge(tlos_conf, conf_data, how='inner', left_index=True, right_index=True, )
qdr_conf = pd.merge(qdr_conf, conf_data, how='inner', left_index=True, right_index=True, )
dist_conf = pd.merge(dist_conf, conf_data, how='inner', left_index=True, right_index=True, )
tcpa_conf = pd.merge(tcpa_conf, conf_data, how='inner', left_index=True, right_index=True, )
dcpa_conf = pd.merge(dcpa_conf, conf_data, how='inner', left_index=True, right_index=True, )
dalt_conf = pd.merge(dalt_conf, conf_data, how='inner', left_index=True, right_index=True, )


#%%Data cration
#Manipulate data LOS
los_data['ownship'] = los_data.index.str[:6]
los_data['intruder'] = los_data.index.str[6:]

#Manipulate data conf

conf_data['conf_start_time'] = tlos_conf.apply(lambda row: row.first_valid_index(), axis=1)
conf_data['conf_end_time'] = tlos_conf.apply(lambda row: row.last_valid_index(), axis=1)
conf_data['conf_time'] = conf_data['conf_end_time']-conf_data['conf_start_time'] 
conf_data['conf_start_angle'] = qdr_conf.iloc[:,:].bfill(axis=1).iloc[:, 0].fillna('unknown')
#conf_data['conf_start_angle'] = abs(abs(conf_data['conf_start_angle_360'] - 180) - 180)

los_data['los_start_time'] = tlos_los.apply(lambda row: row.first_valid_index(), axis=1)
los_data['los_end_time'] = tlos_los.apply(lambda row: row.last_valid_index(), axis=1)
los_data['los_time'] = los_data['los_end_time'] - los_data['los_start_time']
los_data['cpa_time'] = dist_los.idxmin(axis=1)

los_data = pd.merge(los_data, conf_data['conf_time'], how='left', left_index=True, right_index=True)
los_data = pd.merge(los_data, conf_data['conf_start_time'], how='left', left_index=True, right_index=True)
los_data = pd.merge(los_data, conf_data['conf_end_time'], how='left', left_index=True, right_index=True)
los_data['conflict_detection_time'] = los_data['los_start_time'] - conf_data['conf_start_time']

los_data = pd.merge(los_data, create_time, how='left', left_on='ownship', right_index=True)
los_data = pd.merge(los_data, create_time, how='left', left_on='intruder', right_index=True,suffixes=('_ownship', '_intruder'))

los_data = los_data.drop(los_data[(los_data.los_start_time-los_data.spawns_intruder)<min_sep_time].index) #Delete los which spawned at too close together
los_data_del = los_data.drop(los_data[(los_data.los_start_time-los_data.spawns_intruder)>min_sep_time].index) #Delete los which spawned at too close together
los_data = los_data.drop(los_data[los_data.conf_start_time==301].index) #Delete los which spawned at too close together


los_data['conflicts_own'] = 0
los_data['conflicts_int'] = 0
los_data['conflicts_max'] = 0
#Angles
los_data = pd.merge(los_data, conf_data['conf_start_angle'], how='left', left_index=True, right_index=True)
los_data['los_start_angle'] = qdr_los.iloc[:,:].bfill(axis=1).iloc[:, 0].fillna('unknown')
los_data['angle_change'] = abs(los_data['conf_start_angle'] -los_data['los_start_angle'])
los_data['angle_change'] = abs(abs(los_data['angle_change'] - 180) - 180)
#los_data['los_start_angle'] = abs(abs(los_data['los_start_angle'] - 180) - 180)
los_data['cpa_angle'] = 0
         
for i in range(0,len(los_data)):         
    los_data['conflicts_own'][i] = multi_ac_conf2.loc[los_data['conf_time'][i]:los_data['los_start_time'][i], los_data['ownship'][i]].max()
    los_data['conflicts_int'][i] = multi_ac_conf2.loc[los_data['conf_time'][i]:los_data['los_start_time'][i], los_data['intruder'][i]].max()
    los_data['cpa_angle'][i] = qdr_los[los_data['cpa_time'][i]][los_data.index[i]]

#los_data['cpa_angle'] = abs(abs(los_data['cpa_angle'] - 180) - 180)

los_data['conflicts_max'] = los_data[['conflicts_own', 'conflicts_int']].max(axis=1)
los_data['hor_intrusion'] = rpz - dist_los.min(axis=1) #does not work for multi conflicts with the same pair
los_data['hor_severity'] = los_data['hor_intrusion']/rpz

los_data['dalt'] = dalt_los.min(axis=1)
los_data['ver_intrusion'] = hpz - dalt_los.min(axis=1) #does not work for multi conflicts with the same pair
los_data['ver_severity'] = los_data['ver_intrusion']/hpz

#%%Plots
plt.figure(1)
plt.scatter(los_data['conflicts_max'],los_data['hor_severity'])
plt.xlabel('Number of conflicts before los [-]')
plt.ylabel('Horizontal intrusion severity [-]')
print(los_data)

plt.figure(2)
plt.scatter(los_data['conflict_detection_time'],los_data['hor_severity'])
plt.xlabel('Conflict time before LOS [s]')
plt.ylabel('Horizontal intrusion severity [-]')

plt.figure(3)
plt.scatter(los_data['angle_change'],los_data['hor_severity'])
plt.xlabel('angle change [deg]')
plt.ylabel('Horizontal intrusion severity [-]')

plt.figure(4)
los_data['conf_start_angle'] = abs(abs(los_data['conf_start_angle'] - 180) - 180)
plt.scatter(los_data['conf_start_angle'],los_data['hor_severity'])
plt.xlabel('Angle between aircraft at conflict start [deg]')
plt.ylabel('Horizontal intrusion severity [-]')

plt.figure(5)
los_data['los_start_angle'] = abs(abs(los_data['los_start_angle'] - 180) - 180)
plt.scatter(los_data['los_start_angle'],los_data['hor_severity'])
plt.xlabel('Angle between aircraft at LOS start [deg]')
plt.ylabel('Horizontal intrusion severity [-]')

plt.figure(6)
plt.plot(multi_ac_conf2['index'], multi_ac_conf2[multi_ac_conf2.columns[1:]], 'ro')
plt.plot(los_data['conf_start_time'],los_data['conflicts_max'], 'bo')

qdr_conf = qdr_conf.transpose()
qdr_conf = qdr_conf.reset_index()
qdr_los = qdr_los.transpose()
qdr_los = qdr_los.reset_index()
plt.figure(7)
plt.plot(qdr_conf['index'], qdr_conf[qdr_conf.columns[1:]], 'ro')
plt.plot(qdr_los['index'], qdr_los[qdr_los.columns[1:]], 'bo')



plt.show()