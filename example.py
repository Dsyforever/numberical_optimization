# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:42:01 2022

@author: dsy
"""
import numpy as np
import numpy.linalg as LA
from dsy_numberical_optimization import target_function,Optimizer

A=np.mat([[2,0,0],
          [0,3,0],
          [0,0,5]])   
b=np.mat([[4],
          [1],
          [5]])            

def f(x):
    y=0.5*x.T*A*x-b.T*x
    return y
def deri_f(x):
    y=A*x-b
    return y

x=np.mat([[9],
          [9],
          [9]])    
tf=target_function([1,3],f,deri_f)
opt=Optimizer(tf)

x_min=opt.GD_optmize(start=tf.start_point(),A=A,Method="steepest_descent")
print("optmal x:\n{x}".format(x=x_min))
x_min=opt.GD_optmize(start=tf.start_point(),A=A,Method="linear_conjugate_gradient")
print("optmal x:\n{x}".format(x=x_min))
x_min=opt.GD_optmize(start=tf.start_point(),A=A,Method="FR_conjugate_gradient")
print("optmal x:\n{x}".format(x=x_min))
x_min=opt.GD_optmize(start=tf.start_point(),A=A,Method="PR_conjugate_gradient")
print("optmal x:\n{x}".format(x=x_min))
x_min=opt.GD_optmize(start=tf.start_point(),A=A,Method="HR_conjugate_gradient")
print("optmal x:\n{x}".format(x=x_min))


