
'''
This file will output the transformation matrix H.

'''
import numpy as np
import math

out3 = [ 0.0446,  0.0089, -0.2293,  0.2964, -0.2500]

fx = 160 * out3[0]
fy = 120 * out3[1]
c1 = math.cos(out3[2])

c2 = 1
cx = 160 * out3[3]
cy = 120 * out3[4]
s1 = math.sin(out3[2])


H = [[-h*c2/fx, h*s1*s2/fy, h*c2*cx/fx-h*s1*s2*cy/fy-h*c1*s2], 
     [h*s2/fx, h*s1*c2/fy, -h*s2*cx/fx-h*s1*c2*cy/fy-h*c1*c2], 
     [0, h*c1/fy, -h*c1*cy/fy+h*s1], 
     [0, -c1/fy, c1*cy/fy-s1]]


H = np.array(H, dtype = np.float32)

#print H
H = np.reshape(H, (4,3))    
#print H
