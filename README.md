# README

## Welcome to dsy_numberical_optimization

### Introduction 

 

**This is a library of numerical optimization develop by a mathematic student. If you are interest  in this project , welcome to sent your advice to my email:**  madsy@mail.scut.edu.cn.

![book](https://github.com/Dsyforever/numberical_optimization/blob/main/book.jpg)



__To use this project, you need to define your own target function and optimizer with object "target_function" and "Optimizer".__

**Let's introduce some important class first:**

|                  | __class target_function:__                                   |
| ---------------- | ------------------------------------------------------------ |
| --dimension:     | type:__list__                                                |
|                  | if your function is R_n to R, you should input [1,n].        |
| --function:      | type:**function**                                            |
|                  | input your own target funtion which all computational process must base on numpy.mat. |
| --grad_operator: | type:**function**                                            |
|                  | input your own target funtion's grad-operator  which all computational process must base on numpy.mat. |
| --Hessian:       | type:**function**                                            |
|                  | (optional,default=(lambda x: 0))                             |
|                  | input your own target function's Hessian-operator  which all computational process must base on numpy.mat. |



|                    | **class Optimizer:**                                         |
| :----------------- | ------------------------------------------------------------ |
| --target_function: | type:**target_function**                                     |
|                    | input your own target_function.                              |
| --Interpolate:     | type:**str**                                                 |
|                    | (default="bisection",option:"bisection","quadratic")         |
| --error_end:       | type:**float**                                               |
|                    | (default=1e-6)                                               |
|                    | End condition,when \|f(x_k)-f(x_{k-1)}\|< error_end , iteration end. |
| --alpha_logs:      | type:**bool**                                                |
|                    | you can choose to print step_length every step or not.       |
| --steps_logs:      | type:**bool**                                                |
|                    | you can choose print target function's value every step or not. |
| --require_time:    | type:**bool**                                                |
|                    | you can choose print the time spent in the entire optimization process. |

About Optimizer, We will use Optimizer.GD_optmize()  frequently.

|          | **Optimizer.GD_optmize():**                                  |
| -------- | ------------------------------------------------------------ |
| --start: | type:**numpy.mat**                                           |
|          | input your start point                                       |
| --Method | type:**str**                                                 |
|          | (default="steepest_descent",<br />option:"steepest_descent","linear_conjugate_gradient", "FR_conjugate_gradient","PR_conjugate_gradient","HR_conjugate_gradient","SR1","DFP","BFGS","LS_Newton_CG") |
|          | input your optimization Method                               |
| --A      | type:**numpy.mat**                                           |
|          | (optional,default=np.mat([]))                                |
|          | **ps:**Only when you use Method  "linear_conjugate_gradient",you need to Input A. |
|          | A is from funtion form like (0.5x^T *A*x-b^T*x)              |

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
from dsy_numberical_optimization import Optimizer
opt=Optimizer(tf)
```

 Now we can process our optimization(default: steepest_descent Method):

```
x_min=opt.GD_optmize(start=tf.start_point())
print("optmal x:\n{x}".format(x=x_min))
```

result:

```
Method: steepest_descent
whole optmization take 0.010001420974731445s
optmal x:
[[1.99999999]
 [0.33333347]
 [0.99999983]]
```



Example for Method "linear_conjugate_gradient"

```
x_min=opt.GD_optmize(start=tf.start_point(),A=A,Method="linear_conjugate_gradient")
print("optmal x:\n{x}".format(x=x_min))
```

result:

```
Method: linear_conjugate_gradient
waring: this method can only optmize funtion form like (0.5x.T*A*x-b.T*x)
whole optmization take 0.0010037422180175781s
optmal x:
[[2.        ]
 [0.33333333]
 [1.        ]]
```

Example for Method "FR_conjugate_gradient"

```
x_min=opt.GD_optmize(start=tf.start_point(),A=A,Method="FR_conjugate_gradient")
print("optmal x:\n{x}".format(x=x_min))
```

result:

```
Method: FR_conjugate_gradient
whole optmization take 0.039859771728515625s
optmal x:
[[2.00000041]
 [0.33333318]
 [0.99999995]]
```

Example for Method "PR_conjugate_gradient"

```
x_min=opt.GD_optmize(start=tf.start_point(),A=A,Method="PR_conjugate_gradient")
print("optmal x:\n{x}".format(x=x_min))
```

result:

```
Method: PR_conjugate_gradient
whole optmization take 0.02293872833251953s
optmal x:
[[1.99999999]
 [0.33333335]
 [1.00000008]]
```

Example for Method "HR_conjugate_gradient"

```
x_min=opt.GD_optmize(start=tf.start_point(),A=A,Method="HR_conjugate_gradient")
print("optmal x:\n{x}".format(x=x_min))
```

result:

```python
Method: HR_conjugate_gradient
whole optmization take 0.024933576583862305s
optmal x:
[[2.        ]
 [0.33333352]
 [0.99999993]]
```

Example for Method "SR1"

```python
x_min=opt.GD_optmize(start=tf.start_point(),A=A,Method="SR1")
print("optmal x:\n{x}".format(x=x_min))
```

result:

```python
Method: SR1
whole optmization take 0.0010035037994384766s
optmal x:
[[2.        ]
 [0.33333333]
 [1.        ]]
```

Example for Method "DFP"

```python
x_min=opt.GD_optmize(start=tf.start_point(),A=A,Method="DFP")
print("optmal x:\n{x}".format(x=x_min))
```

result:

```python
Method: DFP
whole optmization take 0.002991914749145508s
optmal x:
[[2.        ]
 [0.33333333]
 [1.        ]]
```

Example for Method "BFGS"

```python
x_min=opt.GD_optmize(start=tf.start_point(),A=A,Method="BFGS")
print("optmal x:\n{x}".format(x=x_min))
```

result:

```python
Method: BFGS
whole optmization take 0.00395512580871582s
optmal x:
[[2.        ]
 [0.33333334]
 [1.        ]]
```

Example for Method "HR_conjugate_gradient"

```python
x_min=opt.GD_optmize(start=tf.start_point(),A=A,Method="LS_Newton_CG")
print("optmal x:\n{x}".format(x=x_min))
```

result:

```python
Method: LS_Newton_CG
whole optmization take 0.01695394515991211s
optmal x:
[[1.99999971]
 [0.33333333]
 [0.99999986]]
```

