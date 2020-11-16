import numpy as np
import pandas as pd

test1 = [('1,2'),('2,3'),('3,4')]
test2 = [('A1','A2'),('A2','A3'),('A3','A4')]
test3 =[item[0]+item[1]  for item in test2]

data1 = np.array([1,2,3])
data2 = np.array([4,2,3])
data_all = np.vstack((data1,data2)).T
#print(data1)
df = pd.DataFrame(index=test3,data=data_all) #, columns=['d1','d2']

#test2 = np.array([False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False])

all_data = pd.DataFrame()
for i in range(7):
    los_current = pd.DataFrame(index=test3)
    conflict_current = pd.DataFrame(index=test3,data=data_all)
    all_data = pd.merge(all_data,los_current, how='outer', left_index=True, right_index=True)
    all_data = pd.merge(all_data, conflict_current, how='left', left_index=True, right_index=True,suffixes=('_x', '_y'))






#testcsv = open('testcsv.csv', 'w')
#testcsv.write('{},{}\n'.format("test1 ",test1))
#testcsv.write('{},{}\n'.format("test2 ",test2.replace(',','')))
#testcsv.write('{},{}\n'.format("test3 ",str(list(test3))[1:-1]))

#testcsv.close()


confpairs_list_original = ['AC0052AC0088', 'AC0071AC0084', 'AC0084AC0071', 'AC0088AC0052', 'AC0102AC0103', 'AC0103AC0102']
tLOS = np.array([-71.96147652,  35.37733947,  35.37733947,-71.96147652, 108.83901943, 108.83901943])
simt = 200


tlos_conf_current2 = pd.DataFrame(columns=confpairs_list_original, index=[simt])#, data=[tLOS]
print(tlos_conf_current2)

tlos_conf_current = pd.DataFrame(columns=[confpairs_list_original], index=[simt])
print(tlos_conf_current)