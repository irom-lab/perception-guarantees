import numpy as np
import pickle

k1= 0.1;k2= 0.2
A = np.array([[0,0,1,0],[0,0,0,1],[0,0,-k1,0],[0,0,0,-k2]])
B = np.array([[0,0],[0,0],[k1,0],[0,k2]])
R = np.array([[0.2,0],[0,0.2]])

BRB = B@np.linalg.inv(R)@B.T

file = open('sp_var.pkl','wb')
pickle.dump([k1,k2,A,B,R,BRB],file)