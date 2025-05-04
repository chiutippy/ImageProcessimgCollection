import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, filters
from scipy.interpolate import RectBivariateSpline

'''
Esnake = Integral0~1 [Eint(V(s)) + Eimage(V(s)) + Eext(V(s))] ds

V(s) = (x(s),y(s)) controll points of snake (Parameterized Curves) 

V =[x1,y1,x2,y2...xn,yn] 

'''

'''load image'''
image = color.rgb2gray(data.astronaut())
image = filters.gaussian(image, sigma=2)  # 平滑去除雜訊

'''initial contour'''
s = np.linspace(0, 2*np.pi, 100)
xo = 220 + 100 * np.cos(s)
yo = 100 + 100 * np.sin(s)
snake = np.array([xo, yo]).T

'''gradient of image (sobel)'''
gx, gy = np.gradient(image)
gradient_magnitude = np.sqrt(gx**2 + gy**2)

'''Interpolator, used to look up energy field values'''
## RectBivariateSpline:2D interpolation
interp_gx = RectBivariateSpline(np.arange(image.shape[0]), np.arange(image.shape[1]), -gx)
interp_gy = RectBivariateSpline(np.arange(image.shape[0]), np.arange(image.shape[1]), -gy)

alpha = 0.01    # tension (1st Derivative of Eint)
beta = 20     # curved  (2st Derivative of Eint)
gamma = 0.01    # step
n_points = len(snake)

''' 
Calculate Eint(A) = (alpha * |1st Derivative|**2) + (beta * |2st Derivative|**2) 

1st Derivative ~= Vi+1 - Vi (len of curve)
2st Derivative ~= Vi+1 - 2Vi + Vi-1 ()

'''

# 2st Derivative of Eint
A = np.roll(np.eye(n_points), -1, axis=0) + \
    np.roll(np.eye(n_points), 1, axis=0) - 2 * np.eye(n_points)  
##roll: move array, ex: arr=np.array([1,2,3,4]) => np.roll(a,1) = [4,1,2,3], np.roll(a,-1) = [2,3,4,1]
##eye = identity matrix I (N * N)

#beta * |2st Derivative| "-" alpha * |1st Derivative|,  
#use "-" in order to allow the snake to naturally shrink inwards!!  
#Eint(A)
A = beta * A - alpha * (np.eye(n_points) - np.roll(np.eye(n_points), -1, axis=0))

A = np.linalg.inv(np.eye(n_points) - gamma * A)
## inv: generate inverse matrix(A(-1)) let A.*A(-1) = I 
##Equivalent to solving a system of linear equations

'''
Ftotall = - partial E/ partial V

Use matrix multiplication and interpolation to calculate 
the next moving direction and distance of each control point.

'''
# Iteration of snake
for _ in range(250):
    x, y = snake[:, 0], snake[:, 1]
    
    # Take the image gradient as the external force
    fx = interp_gx.ev(y, x)
    fy = interp_gy.ev(y, x)

    # Update snake point position (internal elasticity + external force push)
    xn = A @ (x + gamma * fx)
    yn = A @ (y + gamma * fy)

    # update snake
    snake = np.stack([xn, yn], axis=1)


plt.imshow(image, cmap='gray')
plt.plot(xo, yo, '--r', label='Initial')
plt.plot(snake[:, 0], snake[:, 1], '-b', label='Final')
plt.legend()
plt.title('Custom Active Contour (Snake)')
plt.axis('off')
plt.show()

