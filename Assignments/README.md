# Finding Local max/min in non-convex multivariate function with Visualization

# Problems
1. Write a pytorch code to find all local max and min values (dy/dx = 0) for the following function
y = 3x-6x3-x4+4 when x = [-6,1]

2. Write a pytorch code to find all local max and min values for the following multivariate function, z = cos(x)sin(y) when x = [-pi,pi] and y=[-pi,pi]

3. Visualization. Plot your numerical solution


## Description of the Solution:

The solution is done in 2 approach for both of the problems. 

This approach contains generating all the pair of points within the given range. Calculate first and second derivate and find which point(s) have the minimum loss value(<= 0.01). The benifit of this process is that we can definately find a minimum value of the loss function but the runtime would be much higher. In this approach the loop takes about 1000+ iterations.

The second approach starts from a random point and then using the gradient descent approach we move forward from that point to find a local minima point. In this approach we can only find one minimum point as the program would exit when the minima is found. To find a single minima this approach takes about 20+ iterations. The only problem is that this solution does not gurantee all the minimum points.


## Presentation 
This repository also contains all the slides that I presented. The MIT Deep Learning Slides and Attention in Deep Learning were imported from external sources. The SQuAD_Project slide was for my final Project.
