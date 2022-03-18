# ReadMe

## Welcome to dsy_numberical_optimization

### Introduction

**This is a library of numerical optimization develop by mathematics student. If you are interest  in this project , welcome to sent your advice to my email:madsy@mail.scut.edu.cn **



![book](C:\Users\dsy\Desktop\ML练习\dsy_numberical_optimization\numberical_optimization\book.jpg)



**To use this project, you need to define your own targer funtion and optimizer with object "target_funtion" and "Optimizer".**



**class target_funtion:**

--dimension:                        type:**list**

​                                               if your $f$ is $R_n \rightarrow  R$, you should input [1,n].

--funtion:                              type:**funtion**

​                                               input your own target funtion which all computational process must base on numpy.mat.

--grad_operator:                  type:**funtion**

​                                               input your own target funtion's grad-operator  which all computational process must base on numpy.mat.

--Hermite:                             type:**funtion**

​                                               (optional,default=(lambda x: 0))

​                                               input your own target funtion's Hermite-operator  which all computational process must base on numpy.mat.

**class Optimizer:**

--target_funtion:                  type:**target_funtion**

​                                               input your own target_funtion.

--Interpolate  :                      type:**str**

​                                               (default="bisection",option:"bisection","quadratic")

--error_end:                          type:**float**

​                                               End condition,when $|f(x_k)-f(x_{k-1)}|<$ error_end , iteration end.

​                                               choose your Interpolation if your Method need to Interpolate.

 --alpha_logs:                        type:**bool**

​                                                you can choose to print step_length every step or not.

--steps_logs:                         type:**bool**

​                                               you can choose print target funtion's value every step or not.

--require_time:                     type:**bool**

​                                                you can choose print the time spent in the entire optimization process.

Introduction about  Optimizer.GD_optmize():

 **Optimizer.GD_optmize():**

--start:                                    type:**numpy.mat**

​                                                input your start point

--Method                                type:**str**

​                                                 (default="steepest_descent",option:"steepest_descent","linear_conjugate_gradient")

​                                                input your optimization Method

--A                                           type:**numpy.mat**

​                                                (optional,default=(lambda x: 0))

​                                                **ps:**Only when you use Method  "linear_conjugate_gradient",you need to Input A.

​                                                A is from funtion form like $(\frac{1}{2}x^T*A*x-b^T*x)$.

First, You need to instantiate the target function using a function and its gradient operator. For example:

```python
import numpy as np
from dsy_numberical_optimization import target_funtion
A=np.mat([[2,0,0],
          [0,3,0],
          [0,0,5]])   
b=np.mat([[4],
          [1],
          [5]])            

def f(x):
    y=0.5*x.T*A*x-b.T*x
    return y
def grad_f(x):
    y=A*x-b
    return y
tf=target_funtion([1,3],f,deri_f)
```

Then, you should  instantiate your Optimizer with your target function.

```python
opt=Optimizer(tf)
```

Then we can process our optimization(default: steepest_descent Method):

```
x_min=opt.GD_optmize(start=tf.start_point())
print("optmal x:\n{x}".format(x=x_min))
```

result:

```
whole optmization take 0.011963844299316406s
optmal x:
[[1.99999999]
 [0.33333333]
 [1.00000036]]
```



Another example for Method "linear_conjugate_gradient"

```
x_min=opt.GD_optmize(start=tf.start_point(),A=A,Method="linear_conjugate_gradient")
print("optmal x:\n{x}".format(x=x_min))
```

result:

```
waring: this method can only optmize funtion form like (0.5x.T*A*x-b.T*x)
whole optmization take 0.0009505748748779297s
optmal x:
[[2.        ]
 [0.33333333]
 [1.        ]]
```

