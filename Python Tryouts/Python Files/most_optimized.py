import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import convolve
from floor_detection import ground_decision_tree
import sys
# file = 'AE4317_2019_datasets/cyberzoo_aggressive_flight/20190121-144646/25181969.jpg'
file_01 = 'AE4317_2019_datasets/cyberzoo_aggressive_flight/20190121-144646/41815146.jpg'
file = 'test.jpg'

assert os.path.exists(file)

im = cv2.imread(file)
sub_width = 25
height = 75

'''
This file is the final python implementation of the simple horizon finder and object detector algorithm

'''

# first create sub_images of original image to create more test samples
sub_images = []
for i in range(sub_width,im.shape[1],2*sub_width):
    sub_images.append(im[240-height:-1,i-sub_width:i+sub_width,:])


### START HERE ###
def find_next(im_bw, prev_points, h,w,step = 3, alpha = .8):
    '''Inputs:
    im_bw: mask of ground, 1 if ground, 0 if not ground
    prev_points: previous points of the horizon, used to find next point more efficiently
    h: height of the sub_image
    step: step size to find next point, if 3 only every three columns, the horizon is detected
    
    output:
    next horizon point or the bottom of the image if no point is found (eg when corner)'''
    # if prev points is only the initial point, dir is [0,step]

    # append prev_points to have better moving average
    prev_points = [prev_points[0],prev_points[0],*prev_points]

    # estimate direction of next point based on previous points
    dir = [int((prev_points[-3][0]*1+prev_points[-2][0]*2+prev_points[-1][0]*3)/6)-prev_points[-1][0],3*(len(prev_points)-2)]
    # if straight edge, we will overshoot and go to end point
    if abs(dir[0]/step)> 2:
        dir[0] = 0

    # initial guess
    init_guess = [prev_points[-1][0] + dir[0], dir[1]]

    edge_found = False
    while not edge_found:

        # if we are not yet on the edge
        if init_guess[0]+1<h:

            # if on edge, pixel above + pixel below == 1 if on ground == 2 if in air == 0
            if (im_bw[init_guess[0], init_guess[1]] + im_bw[init_guess[0]-1, init_guess[1]] ) == 1:
                edge_found = True

            elif (im_bw[init_guess[0], init_guess[1]] + im_bw[init_guess[0]-1, init_guess[1]] ) == 0:
                # its in the air, move down
                init_guess[0] += 1

            else:
                # its on ground, move up
                init_guess[0] -= 1
        else:
            # we are on the edge
            return [h,init_guess[1]]

    return init_guess

import sys
def hor_tracer(sub_im, step = 3):
    ''' input:
    sub_im: the sub_image to trace the horizon
    step: step size of horizon detector

    output:
    hor_points: returns the horizon points found
    '''
    h,w,_ = sub_im.shape

    # first create mask from sub image
    ground = ground_decision_tree(sub_im)[1]
    ground = np.float32(ground)
    ground[ground != 0] = 1
    ground = cv2.convertScaleAbs(ground)

    # start looking for ground
    ground_found = False
    # find first point, and make sure it is not just noise
    i = 0
    col = 0
    while not ground_found and i < (h-5) and col < (w-5):
        if ground[i,col] != 0:
            area = 0
            for j in range(5):
                for k in range(5):
                    area+= ground[i+j,k]
                    
            if area > 20:
                
                ground_found = True
        
        i+=1
        if i == h-5:
            i = 0
            col += 1

    # first point of horizon is at row i
    start_row = i

    # find edge
    hor_points = [[start_row,0]]
    dirs = []
    i = 0
    end = False
    # look for horizon points within width of sub image
    while hor_points[-1][1]<= w-step and i < 100 and not end:
        next_point = find_next(ground, hor_points,h=h, w=w,step = 5)
        hor_points.append(next_point)
        dirs.append(dir)
        # plot the point
        plt.plot(next_point[1], next_point[0],'o',color='y')
        
        i+=1


    # for point in hor_points:
    #     plt.plot(point[1],point[0], 'o', c='y')
    plt.title("Simple horizon finder: Sub Window")
    plt.imshow(sub_im)
    plt.show()
    return hor_points


def second_derivative(hor_points): 
    '''
    Find the second derivatives using the central difference formula,
    f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h^2
    input: 
    hor_points: the collection of horizon points

    output: 
    fdd: second derivative estimate
    y_lst: y values of the hor_points
    '''

    # assume a constant step size
    delta_x = hor_points[1][1] - hor_points[0][1]
    y_lst = []
    for i, point in enumerate(hor_points):
        y_lst.append(point[0])

    fdd_lst = []
    # for all points in y_lst find derivatives
    for i,y in enumerate(y_lst[1:-1]):   
        fdd = (y_lst[i] - 2*y + y_lst[i+2])/(delta_x**2)
        fdd_lst.append(fdd)
    
    return fdd_lst, y_lst

def find_mom(fdd, thr = 1):
    '''
    Find the number of momentum changes with a certain threshold
    input:
    fdd: list of second derivatives
    thr: threshold
    
    output:
    count: the number of momentum changes'''
    count = 0
    for i in fdd:
        if  abs(i) >= thr:
            count+=1
    return count

def confidence(hor_points, im_height):
    '''
    Calculate the confidence of flying straight in our direction
    inputs:
    hor_points: the horizon points from the horizon tracer
    im_height: height of the sub image
    
    outputs:
    conf: confidence of direction as a value between 0 and 240, higher better
    left: true if horizon on the left is higher than on the right
    '''
    y_lst = []
    for i, point in enumerate(hor_points):
        y_lst.append(point[0])
    
    left = True
    y_left = y_lst[0:len(y_lst)//2]
    y_right = y_lst[len(y_lst)//2:]

    if np.average(y_left) < np.average(y_right):
        left = True
    else:
        left = False


    min_y = max(y_lst)
    conf = (im_height-min_y)/im_height
    
    return conf, left


    


plt.figure()
plt.title("Original Image")
plt.imshow(im)
# for sub_image in sub_images[3:6]:
sub_image = sub_images[3]
hor_points = hor_tracer(sub_image)  
fdd, y_lst = second_derivative(hor_points)
print(y_lst)

x_arr = np.arange(0,len(fdd))
print(find_mom(fdd, thr=.5))

print(confidence(hor_points, sub_image.shape[0]))
plt.figure()
plt.title("Second derivative of the horizon line")
plt.plot([0,len(fdd)],[0.5,.5],c='r')
plt.plot([0,len(fdd)],[-0.5,-.5],c='r')
plt.plot(x_arr,fdd)
plt.figure()
plt.title("Original Image")
plt.imshow(im)
plt.show()



### END HERE ###


### SCRAP CODE ###

def find_next(im_bw, prev_points, step = 3, alpha = .8):
    # if prev points is only the initial point, dir is [0,step]
    global O
    # print(prev_points)
    prev_points = [prev_points[0],prev_points[0],*prev_points]
    '''
    # print(prev_points)
    # if len(prev_points) == 1:
    #     dir = [0,step]
    #     init_guess = [prev_points[-1][0] + dir[0], step]
    #     edge_found = False
    #     while not edge_found:
    #         O+=1
    #         # if on edge, pixel above + pixel below == 1 if on ground == 2 if in air == 0
    #         if (im_bw[init_guess[0], init_guess[1]] + im_bw[init_guess[0]-1, init_guess[1]] ) == 1:
    #             edge_found = True

    #         elif (im_bw[init_guess[0], init_guess[1]] + im_bw[init_guess[0]-1, init_guess[1]] ) == 0:
    #             # its in the air
    #             init_guess[0] += 1

    #         else:
    #             # its on ground
    #             init_guess[0] -= 1

    #     return init_guess
    
    # if len(prev_points) == 2:
    #     dir = [int((prev_points[0][0]*2+prev_points[1][0]*3)/5)-prev_points[0][0],6]
    #     init_guess = [prev_points[-1][0] + dir[0], dir[1]]
    #     edge_found = False
    #     while not edge_found:
    #         O+=1
    #         # if on edge, pixel above + pixel below == 1 if on ground == 2 if in air == 0
    #         if (im_bw[init_guess[0], init_guess[1]] + im_bw[init_guess[0]-1, init_guess[1]] ) == 1:
    #             edge_found = True

    #         elif (im_bw[init_guess[0], init_guess[1]] + im_bw[init_guess[0]-1, init_guess[1]] ) == 0:
    #             # its in the air
    #             init_guess[0] += 1

    #         else:
    #             # its on ground
    #             init_guess[0] -= 1

    #     return init_guess
    
    # else:
    # weighted average of last 3 points ((x_-3*1 + x_-2*2+x_-1*3)/6)''' 
    
    dir = [int((prev_points[-2][0]*1+prev_points[-2][0]*2+prev_points[-1][0]*3)/6)-prev_points[-2][0],3*(len(prev_points)-2)]
    init_guess = [prev_points[-1][0] + dir[0], dir[1]]

    edge_found = False
    while not edge_found:
        O+=1
        # if on edge, pixel above + pixel below == 1 if on ground == 2 if in air == 0
        if (im_bw[init_guess[0], init_guess[1]] + im_bw[init_guess[0]-1, init_guess[1]] ) == 1:
            edge_found = True

        elif (im_bw[init_guess[0], init_guess[1]] + im_bw[init_guess[0]-1, init_guess[1]] ) == 0:
            # its in the air
            init_guess[0] += 1

        else:
            # its on ground
            init_guess[0] -= 1

    return init_guess
















def find_mom_changes(horizon_points, rho = 0.6):
    y_lst = []
    for point in horizon_points:
        y_lst.append(point[0])
    S_prev = y_lst[0]
    S_lst = [S_prev]
    S_dir = [S_prev,S_prev,S_prev]
    S_avg_prev = S_prev
    # if abs(S_curr - S_prev) > 3 than 45 degree angle in S plot
    for i,y in enumerate(y_lst[1:-1]):
        # print(S_prev)
        S_curr = rho*S_prev + (1-rho)*y
        S_lst.append(S_curr)
        S_prev = S_curr

        # print(Delta_S)
        # if Delta_S>3:
        #     print('Momentum change detected! Corner found at:', 3*i)
            
         
    return S_lst

















# S_lst = find_mom_changes(hor_points)
# plt.plot(x_arr, S_lst)
# S_lst = find_mom_changes(hor_points, rho = .2)
# x_arr = np.linspace(0,50,len(S_lst))

# x_lst = []
# y_lst = []
# for i, point in enumerate(hor_points):
#     y_lst.append(point[0])
#     x_lst.append(point[1])

# from scipy.interpolate import UnivariateSpline

# y_spl = UnivariateSpline(x_lst,y_lst,s=0,k=4)

# plt.figure()
# plt.semilogy(x_lst,y_lst,'ro',label = 'data')
# x_range = np.linspace(x_lst[0],x_lst[-1],1000)

# plt.semilogy(x_range,y_spl(x_range))
# y_spl_2d = y_spl.derivative(n=2)
# plt.figure()
# plt.plot(x_range,y_spl_2d(x_range))

# plt.plot(x_arr, S_lst)
# fit_lines(hor_points)
# plt.show()



def is_green(sub_im, sensitivity = 2):
    '''Checks number of pixel rows fully green in subimage.'''
    w,h,_ = sub_im.shape

    ground = ground_decision_tree(sub_im)[1]

    

    # ground_edges = convolve(ground,ridge_01)

    # ground_edges[ground_edges!=0] = 1

    ground[ground != 0] = 1
    ground = cv2.convertScaleAbs(ground)

    # plt.imshow(ground)
    # plt.show()


is_green(sub_images[2])