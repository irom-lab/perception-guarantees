import numpy as np
import pickle

# k1= 0.5;k2= 0.5
k1=3.045; k2=2.095; k3=0.03698; k4=0.4645
A = np.array([[0,0,1,0],[0,0,0,1],[0,0,-k1,-k4],[0,0,-k3,-k2]])
B = np.array([[0,0],[0,0],[k1,0],[0,k2]])
R = np.array([[0.5,0],[0,0.5]])

BRB = B@np.linalg.inv(R)@B.T

file = open('sp_var.pkl','wb')
pickle.dump([k1,k2,A,B,R,BRB],file)