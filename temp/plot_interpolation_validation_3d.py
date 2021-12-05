import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D      
from sys import argv
filename = argv[1]
df = pd.read_csv(filename)                                                                                       
                                                                                                                   
fig = plt.figure()                                                                                               
ax = fig.add_subplot(111, projection = '3d')                                                                     
                                                                                                               
x = df['perimeter_dist']                                                                                         
y = df['area']                                                                                                   
z = df['error']                                                                                                  
                                                                                                               
#ax.set_xlabel("Happiness")                                                                                      
#ax.set_ylabel("Economy")                                                                                        
#ax.set_zlabel("Health")                                                                                         
                                                                                                               
ax.scatter(x, y, z)                                                                                              
#plt.savefig(out_fn)
plt.show()
