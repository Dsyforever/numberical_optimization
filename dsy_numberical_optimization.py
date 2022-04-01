# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 21:51:01 2022

@author: dsy
"""

import numpy as np
import numpy.linalg as LA
import time
# import pandas as pd

"""Welcome to dsy_numberical_optimization .To use this project, 
you need to define your own targer funtion and optimizer with object "target_funtion" and "Optimizer".
Those two object are the most important object in this lib.

Here is a introduction for class "target_funtion"

class target_funtion args:
    dimension:
        type:list,if your target funtion is a map R_m to R_1 ,then your dimension=[1,m]
    funtion:
        type:funtion,you should pass your funtion in this arg,such as (lambda x:x.T*x)
    grad_operator:
        type:funtion,you should pass your grad_funtion in this arg,such as(lambda x:2x)
    Hermite:
        type:funtion,you should pass your Hermite_funtion in this arg
        
        
class  Optimizer:
    target_funtion:
        type:target_funtion, your own target funtion.
    Interpolate:
        type:str ,choose one Interpolate,such as quadratic, cubic, or bisection.
    alpha_logs:
        type:bool,you can choose print step_length every step or not.
    steps_logs:
        type:bool,you can choose print target funtion's value every step or not.
    require_time:
        type:bool,you can choose print the time spent in the entire optimization process.





      
class line_search args:
    target_funtion:
        type:target_funtion
    Interpolate:
        type:str Interpolate,such as quadratic, cubic, or bisection.
"""





class target_function:
    def __init__(self,dimension,function,grad_operator,Hessian= lambda x:0):
        self.dimension=dimension
        self.function=function
        self.grad_operator=grad_operator
        self.Hessian=Hessian
    def get_value(self,x):
        return self.function(x)
    def get_grad(self,x):
        return self.grad_operator(x)
    def get_Hessian(self,x):
        return self.Hessian(x)
    def start_point(self):
        rs = np.random.RandomState(1008)
        sp=np.mat(rs.rand(self.dimension[0],self.dimension[1]))
        return sp.T


class Optimizer:
    def __init__(self,target_function,Interpolate="bisection",alpha_logs=False,steps_logs=False,Gly_logs=False,require_time=True,error_end=1e-6):
        self.tf=target_function
        self.Interpolate=Interpolate
        self.error_end=error_end
        self.alpha_logs=alpha_logs
        self.steps_logs=steps_logs
        self.require_time=require_time
        self.Gly_logs=Gly_logs
    def GD_optmize(self,start,Method="steepest_descent",A=np.mat([])):
        x=start
        if Method=="steepest_descent":
            print("Method: steepest_descent")
            ls=line_search(self.tf,self.Interpolate,self.alpha_logs)
        if Method=="linear_conjugate_gradient":
            print("Method: linear_conjugate_gradient")
            print("waring: this method can only optmize funtion form like (0.5x.T*A*x-b.T*x)")
        if Method=="FR_conjugate_gradient":
            print("Method: FR_conjugate_gradient")
            ls=line_search(self.tf,self.Interpolate,self.alpha_logs)
        if Method=="PR_conjugate_gradient":
            print("Method: PR_conjugate_gradient")
            ls=line_search(self.tf,self.Interpolate,self.alpha_logs)
        if Method=="HR_conjugate_gradient":
            print("Method: HR_conjugate_gradient")
            ls=line_search(self.tf,self.Interpolate,self.alpha_logs) 
        begin=time.time()
        x_his=[]
        fy_his=[]
        for i in range(10000):
            x_his.append(i)
            fy_his.append(float(self.tf.get_value(x)))
            #########################################
            if Method=="steepest_descent":
                p=-self.tf.get_grad(x)
                a=ls.search(x,p)  
            ########################################
            if Method=="linear_conjugate_gradient":
                if i==0:
                    r=self.tf.get_grad(x)
                    p=-r
                else:
                    r_next=r+a*A*p
                    beta=float(r_next.T*r_next)/float(r.T*r)
                    p=-r_next+beta*p
                    r=r_next
                a=float(r.T*r)/float(p.T*A*p)
            ##############################################
            if Method=="FR_conjugate_gradient":
                if i==0:
                    p=-self.tf.get_grad(x)
                    a=ls.search(x,p)
                    x_pre=x
                else:
                    beta=float(self.tf.get_grad(x).T*self.tf.get_grad(x))/float(self.tf.get_grad(x_pre).T*self.tf.get_grad(x_pre))
                    p=-self.tf.get_grad(x)+beta*p
                    a=ls.search(x,p)
                    x_pre=x
            ##############################################
            if Method=="PR_conjugate_gradient":
                if i==0:
                    p=-self.tf.get_grad(x)
                    a=ls.search(x,p)
                    x_pre=x
                else:
                    beta=float(self.tf.get_grad(x).T*(self.tf.get_grad(x)-self.tf.get_grad(x_pre)))/float(self.tf.get_grad(x_pre).T*self.tf.get_grad(x_pre))
                    if beta<0:beta=0
                    p=-self.tf.get_grad(x)+beta*p
                    a=ls.search(x,p)
                    x_pre=x
            ################################################
            if Method=="HR_conjugate_gradient":
                if i==0:
                    p=-self.tf.get_grad(x)
                    a=ls.search(x,p)
                    x_pre=x
                else:
                    beta=float(self.tf.get_grad(x).T*(self.tf.get_grad(x)-self.tf.get_grad(x_pre)))/float((self.tf.get_grad(x)-self.tf.get_grad(x_pre)).T*p)
                    if beta<0:beta=0
                    p=-self.tf.get_grad(x)+beta*p
                    a=ls.search(x,p)
                    x_pre=x
                
            x=x+a*p
            if self.steps_logs==True:
                print("{index}th step funtion value:{value} ,x is".format(index=i,value=self.tf.get_value(x)))
                print(x)
            if(LA.norm(self.tf.get_grad(x),2)<self.error_end):break
            if i>9998: print("out of computing capacity")
        end=time.time()
        if self.steps_logs==True or self.require_time==True:
            print("whole optmization take {time}s".format(time=end-begin))
        # if self.Gly_logs==True:
        #     d={'step':x_his,'target_funtion':fy_his}
        #     data=pd.DataFrame(d)
        #     data.plot(x='step',y='target_funtion')
        return x





class line_search:
    def __init__(self,target_funtion,Interpolate,logs=False,c1=1e-4,c2=0.9):
        self.tf=target_funtion
        self.Interpolate=Interpolate
        self.c1=c1
        self.c2=c2
        self.a_pre=0
        self.logs=logs
    def values(self,x,p,a):
        return self.tf.get_value(x+p*a)
    def derivative(self,x,p,a):
        return p.T*self.tf.get_grad(x+p*a)
    def zoom(self,a_l,a_h,x,p):
        if self.Interpolate=="bisection":
            Interpolate=bisection
        if self.Interpolate=="quadratic":
            Interpolate=quadratic
        for i in range(205):
            a=Interpolate(a_l,a_h,self.tf,x,p)
            if ((self.values(x,p,a)>self.values(x,p,0)+self.c1*a*self.derivative(x,p,0)) or
                (self.values(x,p,a)>=self.values(x,p,a_l))):
                a_h=a
            else:
                if((self.derivative(x,p,a))>=self.c2*self.derivative(x,p,0)):
                    return a
                if(self.derivative(x,p,a)*(a_h-a_l)>=0):
                    a_h=a_l
                a_l=a
                   
            if i>200 :print("zoom error ")        
    def search(self,x,p):
        a=0.9
        a_pre=self.a_pre=0
        for i in range(205):
            if((self.values(x,p,a)>self.values(x,p,0)+self.c1*a*self.derivative(x,p,0)) or 
               (i>1 and self.values(x,p,a)>=self.values(x,p,a_pre))) :
                a=self.zoom(a_pre,a,x,p)
                if self.logs== True: print('step length: {alpha}'.format(alpha=a))
                self.a_pre=a
                return a
                
            if ((self.derivative(x,p,a))>=self.c2*self.derivative(x,p,0)):
                if self.logs== True: print('step length: {alpha}'.format(alpha=a))
                self.a_pre=a
                return a
                
            if (self.derivative(x,p,a)>=0):
                a=self.zoom(a,a_pre,x,p)
                if self.logs== True: print('step length: {alpha}'.format(alpha=a))
                self.a_pre=a
                return a
            if i>200 :print("alpha search error ")    
def bisection(a_l,a_h,tf,x,p):
    return (a_l+a_h)/2 
def quadratic(a_l,a_h,tf,x,p):
    a=a_l-((p.T*tf.get_grad(x+p*a_l))*(a_h-a_l)**2)/(2*(tf.get_value(x+p*a_h)-tf.get_value(x+p*a_l)-p.T*tf.get_grad(x+p*a_l)*(a_h-a_l)))
    a=float(a)
    return a
            
            
            
                

