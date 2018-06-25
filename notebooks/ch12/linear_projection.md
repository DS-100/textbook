

```python
# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Image
%matplotlib inline
sns.set()
sns.set_context('talk')
```

# Least Squares Regression as a Projection

Recall that we found the optimal coefficients for linear models by optimizing their cost functions with gradient descent.

We also mentioned that least squares linear regression can be solved analytically. While gradient descent is practical, this geometric perspective will provide a deeper understanding of linear regression.

## Case Study

We've been tasked with finding a good linear model for the below data:

| x | y |
| - |:-|
| 3 | 2 |
| 0 | 1 |
| -1 | -2 |


```python
# HIDDEN
data = pd.DataFrame(
    [
        [3,2],
        [0,1],
        [-1,-2]
    ],
    columns=['x', 'y']
)

sns.regplot(x='x', y='y', data=data, ci=None, fit_reg=False);
```


![png](linear_projection_files/linear_projection_6_0.png)


Assume that the best model is one with the least error, and that the least squared error is an acceptable measure.

Recall that a simple linear regression model follows the form:

$$
\begin{aligned}
f_\hat{\theta} (x_i) 
&= \hat{\theta_0} + \hat{\theta_1} x_i \\
\end{aligned}
$$

Our goal is to find the $\hat{\theta}$ that results in the line with the least squared error $L(\hat{\theta}, \vec{x}, \vec{y})$:

$$
\begin{aligned}
L(\hat{\theta}, \vec{x}, \vec{y})
&= \sum_{i = 1}^{n}(y_i - f_\hat{\theta} (x_i))^2\\
&= \sum_{i = 1}^{n}(y_i - \hat{\theta_0} - \hat{\theta_1} x_i)^2\\
&= \sum_{i = 1}^{n}(y_i - \begin{bmatrix} 1 & x_i \end{bmatrix}
\begin{bmatrix} 
     \hat{\theta_0} \\
     \hat{\theta_1}
\end{bmatrix} ) ^2
\end{aligned}
$$


```python
# HIDDEN
data = pd.DataFrame(
    [
        [3,2],
        [0,1],
        [-1,-2]
    ],
    columns=['x', 'y']
)

sns.regplot(x='x', y='y', data=data, ci=None);
```


![png](linear_projection_files/linear_projection_8_0.png)


In our data set, n = 3, so let's expand out this equation by breaking up the summation.

$$
\begin{aligned}
L(\hat{\theta}, \vec{x}, \vec{y})
&=
(y_1 - \begin{bmatrix} 1 & x_1 \end{bmatrix}
\begin{bmatrix} 
     \hat{\theta_0} \\
     \hat{\theta_1}
\end{bmatrix})^2  \\
&+
(y_2 - \begin{bmatrix} 1 & x_2 \end{bmatrix}
\begin{bmatrix} 
     \hat{\theta_0} \\
     \hat{\theta_1}
\end{bmatrix})^2 \\
&+
(y_3 - \begin{bmatrix} 1 & x_3 \end{bmatrix}
\begin{bmatrix} 
     \hat{\theta_0} \\
     \hat{\theta_1}
\end{bmatrix})^2 \\
\end{aligned}
$$

Notice that our loss function is a sum of squares. Also, notice that the *L2*-norm for a vector has a similar form: 

$$\Vert \vec{v} \Vert = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2}$$ 

If we let $y_i - \begin{bmatrix} 1 & x_i \end{bmatrix}
\begin{bmatrix} 
     \hat{\theta_0} \\
     \hat{\theta_1}
\end{bmatrix} = v_i$: 
$$
\begin{aligned}
L(\hat{\theta}, \vec{x}, \vec{y}) 
&= v_1^2 + v_2^2 + \dots + v_n^2 \\
&= \Vert \vec{v} \Vert^2
\end{aligned}
$$

This means our loss can be expressed as the *L2* norm of some vector $v$, squared. To get the components $v_i$ to equal $y_i - \begin{bmatrix} 1 & x_i \end{bmatrix}
\begin{bmatrix} 
     \hat{\theta_0} \\
     \hat{\theta_1}
\end{bmatrix} \forall i \in [1,3]$:

$$ 
\begin{aligned}
L(\hat{\theta}, \vec{x}, \vec{y})
&= \left \Vert  \qquad   
\begin{bmatrix} y_1 \\ y_2 \\ y_3  \end{bmatrix} \quad - \quad 
\begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ 1 & x_3 \end{bmatrix}
\begin{bmatrix} 
     \hat{\theta_0} \\
     \hat{\theta_1}
\end{bmatrix}
\qquad \right \Vert ^2 \\
&= \left \Vert  \qquad  
\vec{y} 
\quad - \quad 
X 
\begin{bmatrix} 
     \hat{\theta_0} \\
     \hat{\theta_1}
\end{bmatrix}
\qquad \right \Vert ^2 \\
&= \left \Vert  \qquad  
\vec{y} 
\quad - \quad 
\vec{\hat{y}}
\qquad \right \Vert ^2 \\
\end{aligned}
$$

The matrix multiplication $\begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ 1 & x_3 \end{bmatrix}
\begin{bmatrix} 
     \hat{\theta_0} \\
     \hat{\theta_1}
\end{bmatrix}$ is a linear combination of the columns of $X$: each $\hat{\theta_i}$ only ever multiplies with one column of $X$--this perspective shows us that $\hat{y}$ is a linear combination of the features of our data.

This gives us a new perspective on what it means to minimize the least squares error.

$X$ and $\vec{y}$ are fixed, but $\hat{\theta_0}$ and $\hat{\theta_1}$ can take on any value, so $\hat{y}$ can take on any of the infinite linear combinations of the columns of $X$. To have the smallest loss, we want to choose $\hat{\theta_0}$ and $\hat{\theta_1}$ such that the resulting vector is as close to $\vec{y}$ as possible.

## Geometrical Intuition

Now, let's develop an intuition for what it means for the columns of $X$ to span a vector space, and why it matters that $\hat{y}$ is restricted to this vector space.

Let $\vec{x}$ and $\vec{1}$ by two generic vectors, colored black. As we take linear combinations, we will use red for each new vector.


```python
# HIDDEN
plt.quiver([0, 0], [0, 0], [1, 2], [1, -4], 
           angles='xy', scale_units='xy', scale=1, 
           color=['black', 'black'])
plt.xlim(-7, 7)
plt.ylim(-8, 8)
plt.show()
```


![png](linear_projection_files/linear_projection_13_0.png)


When $\hat{\theta_0} = 2$ and $\hat{\theta_1} = 0.5$, we add half of $x$ with two times the unit vector, resulting in a new vector.


```python
# HIDDEN
plt.quiver([0, 0, 0], [0, 0, 0], [1, 2, 3], [1, -4, 0], 
           angles='xy', scale_units='xy', scale=1, 
           color=['black', 'black', 'red'])
plt.xlim(-7, 7)
plt.ylim(-8, 8)
plt.show()
```


![png](linear_projection_files/linear_projection_15_0.png)


When $\hat{\theta_0} = 3$ and $\hat{\theta_1} = -1$, we add $3$ times the unit vector to $-1$ times $x$, resulting in yet another vector.


```python
# HIDDEN
plt.quiver([0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 3, 1], [1, -4, 0, 7], 
           angles='xy', scale_units='xy', scale=1, 
           color=['black', 'black', 'blue', 'red'])
plt.xlim(-7, 7)
plt.ylim(-8, 8)
plt.show()
```


![png](linear_projection_files/linear_projection_17_0.png)


With $\hat{\theta_0} = -2$ and $\hat{\theta_1} = -1$, we get:


```python
# HIDDEN
plt.quiver([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 2, 3, 1, -4], [1, -4, 0, 7, 2], 
           angles='xy', scale_units='xy', scale=1, 
           color=['black', 'black', 'blue', 'blue', 'red'])
plt.xlim(-7, 7)
plt.ylim(-8, 8)
plt.show()
```


![png](linear_projection_files/linear_projection_19_0.png)


With $\hat{\theta_0} = -4$ and $\hat{\theta_1} = 1$, we get:


```python
# HIDDEN
plt.quiver([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 2, 3, 1, -4, -2], [1, -4, 0, 7, 2, -8], 
           angles='xy', scale_units='xy', scale=1, 
           color=['black', 'black', 'blue', 'blue', 'blue', 'red'])
plt.xlim(-7, 7)
plt.ylim(-8, 8)
plt.show()
```


![png](linear_projection_files/linear_projection_21_0.png)


This process can go on and on, showing that when we can use any scalar multiples, we can make an infinite number of linear combinations. However, infinite does not mean any--the linear combinations span a space defined by the original vectors. In this case, we would create a plane as you can start to see above. However, if we had more explanatory variables, the columns of $X$ would create higher dimensional vector spaces.

Formally, this illustrates how vectors $\vec{x}$ and $\vec{1}$ span a vector space.

Back to linear regression, $\hat{y} = X \hat{\theta}$, and is thus the restricted to the vector space spanned by the columns of X. Let's look at the significance of these facts.

As a reminder, here is our loss function and scatter plot:

$$L(\hat{\theta}, \vec{x}, \vec{y}) \quad = \quad \left \Vert  \quad  
\vec{y} 
\quad - \quad 
X \hat{\theta}
\quad \right \Vert ^2$$


```python
# HIDDEN
sns.regplot(x='x', y='y', data=data, ci=None, fit_reg=False);
```


![png](linear_projection_files/linear_projection_24_0.png)


By inspection, we see that no line can perfectly fit our points, so we cannot achieve 0 loss. Thus, we know that $\vec{y}$ is not in the plane spanned by $\vec{x}$ and $\vec{1}$, represented as a parallelogram below.

<img src="../../notebooks-images/linear_projection1.png" width="500" />

Since our error is based on distance, we can see that to minimize $ L(\hat{\theta}, \vec{x}, \vec{y}) = \left \Vert  \vec{y} - \hat{y} \right \Vert ^2$, we want $\hat{y}$ to be as close to $\vec{y}$ as possible.

Mathematically, we are looking for the projection of $\vec{y}$ onto the vector space spanned by the columns of $X$, because $\hat{y}$ must be inside the vector space, and the projection of $\vec{y}$ has the property of being the closest point in the vector space to the vector $y$. Thus, choosing $\hat{\theta}$ such that $X \hat{\theta} = \hat{y} = proj_{spanX} \quad y$ is the best solution.
<img src="../../notebooks-images/linear_projection2.png" width="500" />

To see why, consider other points on the vector space, in purple.
<img src="../../notebooks-images/linear_projection3.png" width="500" />

By the Pythagorean Theorem, any other point on the plane is farther from $\vec{y}$ than $\hat{y}$ is. The length of the perpendicular corresponding to $\hat{y}$ represents the least squared error.

## Linear Algebra

Since we've snuck in a lot of linear algebra concepts already, all that's left is solving for the $\hat{\theta}$ that yields our desired $\hat{y}$.

A couple things to note: 

<img src="../../notebooks-images/linear_projection5.png" width="500" />

- $\hat{y} + e = \vec{y}$
- $\vec{e}$ is perpendicular to $\vec{x}$ and $\vec{1}$
- $\hat{y} = X \hat{\theta^*}$ is the vector closest to $y$ in the vector space spanned by $\vec{x}$ and $\vec{1}$

Thus, we arrive at the equation:

$$X  \hat{\theta^*} + \vec{e} = \vec{y}$$

Left-multiplying both sides by $X^T$:

$$X^T X  \hat{\theta^*} + X^T \vec{e} = X^T \vec{y}$$

Since $\vec{e}$ is perpendicular to the columns of $X$, $X^T \vec{e}$ is a column vector of $0$'s. Thus, we arrive at the Normal Equation:

$$X^T X  \hat{\theta^*} = X^T \vec{y}$$

From here, we can easily solve for $\hat{\theta^*}$ by left-multiplying both sides by $(X^T X)^{-1}$:

$$\hat{\theta^*} = (X^T X)^{-1} X^T \vec{y}$$

## Finishing up the Case Study

Let's return to our case study and apply what we've learned, and explain why our solution is sound

$$
\vec{y} = \begin{bmatrix} 2 \\ 1 \\ -2  \end{bmatrix} \qquad X = \begin{bmatrix} 1 & 3 \\ 1 & 0 \\ 1 & -1 \end{bmatrix}
$$


$$
\begin{align}
\hat{\theta^*} 
&= 
\left(
\begin{bmatrix} 1 & 1 & 1 \\ 3 & 0 & -1 \end{bmatrix}
\begin{bmatrix} 1 & 3 \\ 1 & 0 \\ 1 & -1 \end{bmatrix}
\right)^{-1}
\begin{bmatrix} 1 & 1 & 1 \\ 3 & 0 & -1 \end{bmatrix}
\begin{bmatrix} 2 \\ 1 \\ -2  \end{bmatrix} \\
&= 
\left(
\begin{bmatrix} 3 & 2\\ 2 & 10 \end{bmatrix}
\right)^{-1}
\begin{bmatrix} 1 \\ 8 \end{bmatrix} \\
&=
\frac{1}{30-4}
\begin{bmatrix} 10 & -2\\ -2 & 3 \end{bmatrix}
\begin{bmatrix} 1 \\ 8 \end{bmatrix} \\
&=
\frac{1}{26}
\begin{bmatrix} -6 \\ 22 \end{bmatrix}\\
&=
\begin{bmatrix} - \frac{3}{13} \\ \frac{11}{13} \end{bmatrix}
\end{align}
$$

We have analytically found that best model for least squares regression is $f_\hat{\theta} (x_i) = - \frac{3}{13} + \frac{11}{13} x_i$. We know that our choice of $\hat{\theta}$ is sound by the mathematical property that the projection of $\vec{y}$ onto the span of the columns of $X$ yields the closest point in the vector space to $\vec{y}$. Under linear constraints using the least squares metric, solving for $\hat{\theta}$ by taking the projection guarantees us the optimal solution.
