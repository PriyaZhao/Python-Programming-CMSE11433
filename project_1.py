#!/usr/bin/env python
# coding: utf-8

# # Project 1 - Quality control for a clock manufacturing company [40 marks]
# 
# ---
# 
# Make sure you read the instructions in `README.md` before starting! In particular, make sure your code is well-commented, with sensible structure, and easy to read throughout your notebook.
# 
# ---
# 
# There is an imaginary clock manufacturing company that wants you to develop software to check the quality of its products. The clocks produced by this company have **two hands**:
# 
# - the small hand is **red** and indicates the **hour**,
# - the long hand is **green** and indicates the minutes.
# 
# We refer to these as *the hour hand* and *the minute hand* respectively. These clocks do not have any other hands (although some other clocks have a third hand indicating the seconds).
# 
# It is very important for these hands to be properly aligned. For example, if the hour hand is pointing to the hour `3` (being horizontal and pointing toward right), the minute hand should be pointing toward the hour `12` (vertical and pointing upward). Another example is when the hour hand is pointing to the hour `1:30` (making a 45 degree angle from the vertical line), the minute hand should be pointing toward hour `6` (vertical and downward).
# 
# | Correct `1:30`, the hour hand is halfway between 1 and 2. | Incorrect `1.30`, the hour hand is too close to 1. |
# |:--:|:--:|
# | ![Correct 1.30](graphics/one_thirty_correct.png) | ![Incorrect 1.30](graphics/one_thirty_incorrect.png) |
# 
# Due to production imprecisions, this is not the case all the time. Your software package will **quantify the potential misalignments** and help the company to return the faulty clocks back to the production line for re-adjustment.
# 
# You will achieve this goal in several steps during this project. Most steps can be done independently. Therefore, if you are struggling with one part, you can move on to other tasks and gain the marks allocated to them.
# 
# For most tasks, under "✅ *Testing:*", you will be given instructions on how to check that your function works as it should, even if you haven't done the previous task.
# 
# 
# ---
# 
# ## Task 1: Reading images into NumPy arrays [3 marks]
# 
# The company takes a picture of each clock, and saves it as a PNG image of 101x101 pixels. The folder `clock_images` contains the photos of all the clocks you need to control today.
# 
# In a PNG colour image, the colour of each pixel can be represented by 3 numbers between 0 and 1, indicating respectively the amount of **red**, the amount of **green**, and the amount of **blue** needed to make this colour. This is why we refer to colour images as **RGB** images.
# 
# - If all 3 values are 0, the pixel is black.
# - If all 3 values are 1, the pixel is white.
# - If all 3 values are the same, the pixel is grey. The smaller the values, the darker it is.
# - Different amounts of red, green, and blue correspond to different colours.
# 
# For example, select a few colours [using this tool](https://doc.instantreality.org/tools/color_calculator/), and check the RGB values for that colour in the *RGB Normalized decimal* box. You should see that, for instance, to make yellow, we need a high value of red, a high value of green, and a low value of blue.
# 
# If you'd like more information, [this page](https://web.stanford.edu/class/cs101/image-1-introduction.html) presents a good summary about RGB images.
# 
# ---
# 
# 🚩 Study the [documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html) for the functions `imread()` and `imshow()` from `matplotlib.pyplot`. Then, write code below to read the `clock_0` image from `batch_0` into a NumPy array, and display the image.
# 
# You will obtain a NumPy array with shape `(101, 101, 3)`, i.e. an array which is 3 layers deep. Each of these layers is a 101x101 array, where the elements represent the intensity of red, green, and blue respectively, for each pixel. For example, the element of this array with index `[40, 20, 2]` corresponds to the amount of blue in the pixel located in row 40, column 20.
# 
# Create a second figure, with 3 sets of axes, and use `imshow()` to display each layer separately on its own set of axes. Label your figures appropriately to clearly indicate what each image is showing.
# 
# *Note: you can use `ax.imshow()` to display an image on the axes `ax`, the same way we use `ax.plot()`.*

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

# get the RGB array of clock_0 from batch_0
RGB_c0_b0 = plt.imread('clock_images/batch_0/clock_0.png')

# create a figure to display the clock image
plt.imshow(RGB_c0_b0)
plt.title('Clock_0, Batch_0') 
plt.axis('off')

# transfer the initial array into the corresponding RGB array of each layer respectively
# For example: to display the layer corresponds to the amount of red, 
# we only need the first number which indicates red and let the other two number be zero.
Red_c0_b0 = np.copy(RGB_c0_b0)
Red_c0_b0[:,:,1] = np.zeros(Red_c0_b0[:,:,1].shape)
Red_c0_b0[:,:,2] = np.zeros(Red_c0_b0[:,:,2].shape)
Green_c0_b0 = np.copy(RGB_c0_b0)
Green_c0_b0[:,:,0] = np.zeros(Green_c0_b0[:,:,0].shape)
Green_c0_b0[:,:,2] = np.zeros(Green_c0_b0[:,:,2].shape)
Blue_c0_b0 = np.copy(RGB_c0_b0)
Blue_c0_b0[:,:,0] = np.zeros(Blue_c0_b0[:,:,0].shape)
Blue_c0_b0[:,:,1] = np.zeros(Blue_c0_b0[:,:,1].shape)

# create the second figure with 3 plots to display each layer separately
fig, ax = plt.subplots(1,3)
color_list = ['red','green','blue']
array_list = [Red_c0_b0,Green_c0_b0,Blue_c0_b0]
for i in range(3):
    ax[i].imshow(array_list[i])
    ax[i].set_title(color_list[i]+' layer')
    ax[i].axis('off')


# ---
# ## Task 2: Clean up the images to extract data [6 marks]
# 
# Later in Task 3, we will use **linear regression** to find the exact position of both clock hands. To perform linear regression, we will need the **coordinates of the pixels** belonging to each hand; then, we will be able to fit a line through these pixels.
# 
# This task is concerned with extracting the correct pixel coordinates from the image.
# 
# ---
# 
# 🚩 Write a function `get_clock_hands(clock_RGB)`, which takes one input argument `clock_RGB`, a NumPy array of size 101x101x3 representing an RGB image of a clock, and returns 2 NumPy arrays with 2 columns each, such that:
# 
# - In the first array, each row corresponds to the `[row, column]` index of a pixel belonging to the **hour hand**.
# - In the second array, each row corresponds to the `[row, column]` index of a pixel belonging the **minute hand**.
# 
# The goal is to obtain, for each hand, a collection of `[row, column]` coordinates which indicate where on the picture is the clock hand. You will need to figure out a way to decide whether a given pixel belongs to the hour hand, the minute hand, or neither.
# 
# 
# ---
# 
# ***Important note:*** the pictures all contain some amount of noise and blur. Depending on how you decide to count a certain pixel or not as part of a clock hand, your function will pick up different pixels. There isn't just one possible set of pixel coordinates to pick up for a given image -- the most important thing is that the pixels you extract **only** belong to one of the two hands, and not to the background for example. This will ensure that you can use linear regression efficiently.
# 
# ---
# 
# ✅ *Testing:* For example, for the tiny 7x7 clock below (a 7x7 pixel image is available in the `testing` folder for you to try your function):
# 
# | Clock | Hour hand | Minute hand |
# |:--:|:--:|:--:|
# | <img src="graphics/task2.png" alt="Task 2 example" style="width: 100px;"/> | [[1, 1]<br/> [2, 2]] | [[3, 3]<br/> [4, 3]<br/> [4, 4]<br/> [5, 4]<br/> [6, 5]] |

# In[3]:


def get_clock_hands(clock_RGB):
    '''
    Extract a collection of [row, column] coordinates for each hand which indicate where on the picture is the clock hand.
    
    Input: clock_RGB (array): a NumPy array of size 101x101x3 representing an RGB image of a clock.
    
    Output: hour_hand_coord (array): a NumPy array contains all the indexes of pixels belonging to the hour hand.
            minute_hand_coord (array): a NumPy array contains all the indexes of pixels belonging to the minute hand.
    '''
    
    # create two empty lists to store the index of each hand that the function extracts later
    hour_hand_index = []
    minute_hand_index = []
    # get the length of row and column of the RGB array
    len_row = clock_RGB.shape[0]     
    len_col = clock_RGB.shape[1]
    
    # check whether each pixel belongs to the hour hand or minute hand
    for i in range(len_row):
        for j in range(len_col):
            RGB_pixel = clock_RGB[i,j,:]
            
            # If the pixel is visually red, then it belongs to the hour hand
            # the specific criteria was obtained by analyzing 10 random sampling red points from clock_1 of batch_0
            if (RGB_pixel[0] > 0.75) and (RGB_pixel[1] < 0.55):
                hour_hand_index.append([i,j])
            
            # If the pixel is visually green, then it belongs to the minute hand
            # the specific criteria was obtained by analyzing 10 random sampling green points from clock_1 of batch_0
            elif (RGB_pixel[1] > 0.75) and (RGB_pixel[1] - RGB_pixel[0] > 0.1):
                minute_hand_index.append([i,j])
    
    # transfer the obtained indexes into numpy array and then return
    hour_hand_coord = np.array(hour_hand_index)
    minute_hand_coord = np.array(minute_hand_index)
    
    return hour_hand_coord, minute_hand_coord


# two tests below
# test 1   
# test_array = plt.imread('testing/task2_7x7.png')
# t1, t2 = get_clock_hands(test_array)
# print(t1)
# print(t2)

# test 2: Check if the coordinates found by the function match the original image
# test_array = plt.imread('clock_images/batch_0/clock_27.png')
# plt.imshow(test_array)
# t1, t2 = get_clock_hands(test_array)
# fig,ax = plt.subplots(figsize = [3.9,3.9])
# ax.plot(t1[:,1],t1[:,0],'r-')
# ax.plot(t2[:,1],t2[:,0],'g-')
# ax.set_xlim(0,102)
# ax.set_ylim(0,102)
# ax.invert_yaxis()
# plt.show()


# ---
# 
# ## Task 3: Calculate the angle of the two hands [9 marks]
# 
# Now that we have pixel locations for each hand, we can estimate the **angle** between each hand and the 12 o'clock position. We will use this angle later to determine the time indicated by each hand. For instance, the figure below shows the angle made by the hour hand with the 12 o'clock position.
# 
# ![Angle between hour hand and 12 o'clock](graphics/angle.png)
# 
# ---
# 
# 🚩 Write a function `get_angle(coords)` which takes one input argument, a NumPy array with 2 columns representing `[row, column]` pixel coordinates of one clock hand, exactly like one of the arrays returned by `get_clock_hands()` from Task 2.
# 
# - Your function should use these pixel coordinates to find a **line of best fit** using linear regression.
# - Then, using this line of best fit, you should determine and **return** the angle between the clock hand and the 12 o'clock position, measured in **radians**.
# 
# The angle should take a value between $0$ (inclusive) and $2\pi$ (exclusive) radians, where $0\, \text{rad}$ corresponds to the 12 o'clock position.
# 
# ---
# 
# ***Notes:***
# 
# - When performing linear regression, you will need to pay particular attention to the case where the clock hand is vertical or almost vertical.
# - Beware of the correspondance between `[row, column]` index and `(x, y)` coordinate for a given pixel.
# - Note that the meeting point of the 2 clock hands may not be exactly at `[50, 50]`. Some of the pictures have a small offset.
# - Partial attempts will receive partial marks. For instance, if you are struggling with using linear regression, or if you don't know how to account for possible offset of the centre, you may receive partial marks if you use a simpler (but perhaps less accurate) method.
# 
# ---
# 
# ✅ *Testing:* the files `task3_hourhand.txt` and `task3_minutehand.txt` are provided for you to test your function in the `testing` folder. Use `np.loadtxt()` to read them.
# 
# With these coordinates, you should find an angle of approximately 4.2 radians for the hour hand, and 5.7 radians for the minute hand.

# In[4]:


import math

def get_angle(coords):
    '''
    Estimate the angle between the hand and the 12 o'clock position according to the coordinates that belongs to it.
    
    Input: coords (array): a NumPy array with 2 columns representing [row, column] pixel coordinates of one clock hand.
    
    Output: r (float): the angle between the clock hand and the 12 o'clock position, 
                       measured in radians (between 0(inclusive) and 2𝜋(exclusive)).
    '''
        
    # do some pre-processing on the input coordinates before linear regression
    # get the minimum and maximum x coordinate of the hand
    xmin = int(np.amin(coords[:,1]))
    xmax = int(np.amax(coords[:,1]))
    # create two empty lists to store the new coordinates after pre-processing
    n = coords.shape[0]                # the amount of input coordinates
    x = []
    y = []
    # take the average of the coordinate points that have the same x value
    for i in range(xmin,xmax+1):
        y_sum = 0
        count = 0
        for j in range(n):
            if coords[j,1] == i:       # check if the coordinate has the same x value
                y_sum += coords[j,0]
                count += 1
        if count != 0:
            y_bar = y_sum / count      # take the average
            y.append(y_bar)
            x.append(i)
        
    # perform linear regression to obtain the slope and raw radian
    A = np.vstack([np.array(x), np.ones(len(x))]).T
    s, b = np.linalg.lstsq(A, np.array(y), rcond=None)[0]
    r0 = math.atan(s)

    # divide the clock into four zones and check which zone the hand is in
    # because the meeting point of the 2 clock hands may not be exactly at [50, 50],
    # we use the midpoint of the hand to check which zone the hand is in
    ymin = np.amin(y)
    ymax = np.amax(y)
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2

    # the conversion formula of raw radian to real radian (angle) is different for diffenrent zones
    # In particular: when in zone 1 or 3: r=r0+3/2pi, when in zone 2 or 4: r=r0+1/2pi
    # And then we found that we could just consider two conditions according to the x coordinate
    # But first we should consider the vertical case 
    if len(x) == 1:                  # the clock hand is vertical         
        if ymid < 50:                # the clock hand is up
            r = 0
        else:                        # the clock hand is down
            r = math.pi
    else:                            # the clock hand is not vertical 
        if xmid < 50:                # the clock hand is in zone 1 or 3
            r = r0 + 3/2 * math.pi        
        else:                        # the clock hand is in zone 2 or 4
            r = r0 + 1/2 * math.pi
    
    return r

# two tests below
# test 1 
# test_coords_hourhand = np.loadtxt('testing/task3_hourhand.txt')
# print(get_angle(test_coords_hourhand))
# test_coords_minutehand = np.loadtxt('testing/task3_minutehand.txt')
# print(get_angle(test_coords_minutehand))

# test 2: check the angle of an image
# test_array = plt.imread('clock_images/batch_1/clock_17.png')
# test_coords_hourhand, test_coords_minutehand = get_clock_hands(test_array)
# print(get_angle(test_coords_hourhand))
# print(get_angle(test_coords_minutehand))


# ---
# 
# ## Task 4: Visualising the clock [6 marks]
# 
# 🚩  Use `matplotlib` and your artistic skills to visualise the clock. Write a function `draw_clock(angle_hour, angle_minute)` that takes 2 input arguments, corresponding to the two angles of the clock hands, and draw a clock with the precise location of both hands.
# 
# Your plot may include the number associated to hours, a background like a circle, an arrow head for each hand etc.
# 
# ---
# 
# ✅ *Testing:* with `angle_hour` set to $\frac{\pi}{3}$ and `angle_minute` set to $\frac{11\pi}{6}$, the hour hand should point exactly at 2, and the minute hand should point exactly at 11.
# 
# There is also an example image in the `testing` folder, which was produced entirely with `matplotlib`. This is just to give you an idea of what is possible to do -- you shouldn't attempt to reproduce this particular example, don't hesitate to get creative!

# In[4]:


from matplotlib.patches import Circle
get_ipython().run_line_magic('matplotlib', 'inline')

def draw_clock(angle_hour, angle_minute):
    '''
    Visualise a clock given the angles of its hour and minute hand.
    
    Input: angle_hour (float): the angle between the hour hand and the 12 o'clock position, measured in radians.
           angle_minute (float): the angle between the minute hand and the 12 o'clock position, measured in radians.
    
    Output: a plot that displays a clock with the precise location of both hands.
    '''
    
    # create a figure and draw a square outer frame line
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    
    # add 3 circles to the background
    circle1 = Circle(xy=(0.5,0.5), radius = 0.45, fc='none',ec='k', linewidth=2)
    circle2 = Circle(xy=(0.5,0.5), radius = 0.43, fc='none',ec='k', linewidth=1)
    circle3 = Circle(xy=(0.5,0.5), radius = 0.3, fc='none',ec='k',ls = '-.', linewidth=3)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    
    # add the number 
    number = [1,2,3,4,5,6,7,8,9,10,11,12]
    r_number = 0.35                                          # The distance between mark number and center(radius)
    x,y = (0.5,0.5)                                          # The coordinates of the center point
    for i in number:
        txt_angle = i * (np.pi/6)
        txt_x = x + r_number * np.sin(txt_angle)
        txt_y = y + r_number * np.cos(txt_angle)
        plt.text(txt_x,txt_y,str(i),ha='center', va='center',fontsize=20,fontweight='bold')
        
    # draw the hands as arrows
    r1,r2 = (0.2,0.3)                                        # The length of the hour and minute hands
    arrowlen_hour_x = r1 * np.sin(angle_hour)                # The corresponding x and y length
    arrowlen_hour_y = r1 * np.cos(angle_hour)
    arrowlen_minute_x = r2 * np.sin(angle_minute)
    arrowlen_minute_y = r2 * np.cos(angle_minute)
    ax.arrow(x,y,arrowlen_hour_x,arrowlen_hour_y,width=0.018,length_includes_head=True,head_width=0.03,head_length=0.03,ec='k',fc='r')
    ax.arrow(x,y,arrowlen_minute_x,arrowlen_minute_y,width=0.01,length_includes_head=True,head_length=0.03,ec='k',fc='g')
    
# test code
# draw_clock(np.pi/3, np.pi*11/6)
# help(draw_clock)


# ---
# ## Task 5: Analog to digital conversion [5 marks]
# 
# 🚩 Write a function `analog_to_digital(angle_hour, angle_minute)` that takes two input arguments, corresponding to the angles formed by each hand with 12 o'clock, and returns the time in digital format. More specifically, the output is a string showing the time in hour and minute in the format `hh:mm`, where `hh` is the hour and `mm` is the minute.
# 
# - When the hour is smaller than 10, add a leading zero (e.g. `04:30`).
# - When the hour is zero, display `12`.
# 
# At this point, your function is not concerned about the imprecision. It should calculate the hour from the hour hand, and the minute from the minute hand, separately.
# 
# ---
# ✅ *Testing:* the same angles as in Task 4 should give you `02:55`.

# In[6]:


def analog_to_digital(angle_hour, angle_minute):
    '''
    Display the time in digital format according to the angles of hour and minute hand.
    
    Input: angle_hour (float): the angle between the hour hand and the 12 o'clock position, measured in radians.
           angle_minute (float): the angle between the minute hand and the 12 o'clock position, measured in radians.
    
    Output: time (str): the time showing the hour and minute in the format hh:mm.
    '''
    
    # Calculate the hour from the hour hand and the minute from the minute hand
    # For every 1/6pi increases in the angle of hour hand, the time increases by one hour. 
    # Also, for every 1/30pi increases in the angle of minute hand, the time increases by one minute
    # Because of the nature of a clock, we use 'int' for the hour hand, and use 'round' for the minute hand.
    time_hour = int(angle_hour / (np.pi/6))
    time_minute = round(angle_minute / (np.pi/30))
    
    # modifications for some special cases
    if time_hour == 0:
        time_hour = 12
    if time_hour < 10:
        time_hour = '0' + str(time_hour)
    if time_minute < 10:
        time_minute = '0' + str(time_minute) 
        
    # create the time as a string in the format hh:mm    
    time = str(time_hour) + ':' + str(time_minute)
    
    return time

# tests
# print(analog_to_digital(np.pi/3, np.pi*11/6))
# print(analog_to_digital(get_angle(test_coords_hourhand),get_angle(test_coords_minutehand)))
# help(analog_to_digital)


# ---
# ## Task 6: Find the misalignment [5 marks]
# 
# Now that you have extracted useful information from the pictures, you need to check if the two hands are aligned properly. To do so, you will need to find the exact time that the **small hand** is showing, in hours and minutes. Then, compare with the minutes that the big hand is showing, and report the difference.
# 
# Note that the misalignment will never be more than 30 minutes. For example, if you read a 45-minute difference between the minutes indicated by the hour hand and by the minute hand, you can realign the minute hand by 15 minutes in the other direction instead.
# 
# ---
# 
# 🚩 Write a function `check_alignment(angle_hour, angle_minute)` which returns the misalignment in minutes.
# 
# Make sure you structure you code sensibly. You may wish to use some intermediate functions to do the sub-tasks.
# 
# ---
# ✅ *Testing:* the same angles as in Task 4 should give you a 5-minute misalignment.

# In[7]:


def checktime_byhourhand(angle_hour):
    '''
    Find the exact time that the hour hand is showing, in hours and minutes.
    
    Input: angle_hour (float): the angle between the hour hand and the 12 o'clock position, measured in radians.
    
    Output: time_hour (float): the time in hours showing by the hour hand.
            time_minute (float): the time in minutes showing by the hour hand.
    '''
    
    # Define the relationship between time and angle
    onehour = np.pi/6
    oneminute = np.pi/360
    
    # Calculate the exact time showed by the hour hand according to its corresponding time and angle relationship
    # For minute, we should calculate the time at which the hand is closest, so we use 'round'
    time_hour = angle_hour // onehour
    time_minute = round((angle_hour % onehour) / oneminute)  
    
    return time_hour,time_minute

def checktime_byminutehand(angle_minute):
    '''
    Find the exact time in minutes that the minute hand is showing.
    
    Input: angle_minute (float): the angle between the minute hand and the 12 o'clock position, measured in radians.
    
    Output: time_minute (float): the time in minutes showing by the minute hand.
    '''
    
    # Define the relationship between time and angle
    oneminute = np.pi/30
    
    # Calculate the exact time showed by the minute hand according to its corresponding time and angle relationship
    time_minute = round(angle_minute / oneminute)
    
    return time_minute

def check_alignment(angle_hour, angle_minute):
    '''
    Report the misalignment in minutes indicated by the hour hand and by the minute hand.
    
    Input: angle_hour (float): the angle between the hour hand and the 12 o'clock position, measured in radians.
           angle_minute (float): the angle between the minute hand and the 12 o'clock position, measured in radians.
    
    Output: misalignment (int): the misalignment in minutes.
    '''
    
    # Get the exact time showed by the hour and minute hands
    minute_byhourhand = checktime_byhourhand(angle_hour)[1]
    minute_byminutehand = checktime_byminutehand(angle_minute)
    
    # Calculate the difference and transfer it into the misalignment
    time_diff = abs(minute_byminutehand - minute_byhourhand)
    misalignment = 30 - abs(time_diff - 30)     # make sure the misalignment is more than 30 minutes
    
    return misalignment

# test
# print(check_alignment(np.pi/3, np.pi*11/6))
# print(type(check_alignment(np.pi/3, np.pi*11/6)))
# help(checktime_byhourhand)
# help(checktime_byminutehand)
# help(check_alignment)


# ---
# ## Task 7: Putting it all together [6 marks]
# 
# Now that you have successfully broken down the problem into a sequence of sub-tasks, you need to combine all the above steps in one function.
# 
# 🚩 Write a function `validate_clock(filename)` that takes the name of an image file (a picture of a clock face) as an input argument, and returns the misalignment in minutes as an integer.
# 
# Then, write a function `validate_batch(path, tolerance)` which takes 2 input arguments: `path`, a string to indicate the path of a folder containing a batch of clock pictures, and `tolerance`, a positive integer representing the maximum tolerable number of minutes of misalignment for a clock to pass the quality control check.
# 
# Your `validate_batch()` function should write a .txt file called `batch_X_QC.txt` (where `X` should be replaced by the batch number), containing the following information:
# 
# ```
# Batch number: [X]
# Checked on [date and time]
# 
# Total number of clocks: [X]
# Number of clocks passing quality control ([X]-minute tolerance): [X]
# Batch quality: [X]%
# 
# Clocks to send back for readjustment:
# clock_[X]   [X]min
# clock_[X]   [X]min
# clock_[X]   [X]min
# [etc.]
# ```
# 
# The square brackets indicate information which you need to fill in.
# 
# - You will need to check all pictures in the given folder. You may wish to use Python's `os` module.
# - The date and time should be the exact date and time at which you performed the validation, in the format `YYYY-MM-DD, hh:mm:ss`. You may wish to use Python's `datetime` module.
# - The batch quality is the percentage of clocks which passed the quality control in the batch, rounded to 1 decimal place. For example, in a batch of 20 clocks, if 15 passed the control and 5 failed, the batch quality is `75.0%`.
# - List all clock numbers which should be sent back for realignment, in **decreasing order of misalignment**. That is, the most misaligned clock should appear first.
# - The list of clocks to send back and the misalignment in minutes should be vertically aligned, in a way which makes the report easy to read. Check the example in the `testing` folder.
# - Your function should not return anything, simply write the .txt report.
# 
# For instance, to use your function to check batch 1 with a 2-minute maximum acceptable misalignment, the command will be `validate_batch('clock_images/batch_1', 2)`.
# 
# ---
# 
# ✅ *Testing:* There is an example report in the `testing` folder (for a batch which you do not have), to check that your report is formatted correctly.
# 
# ---
# 
# 🚩 Use your function `validate_batch()` to generate quality control reports for the 5 batches of clocks provided in the `clock_images` folder, with a tolerance of 3 minutes.
# 
# Your reports should all be saved in a folder called `QC_reports`, which you should create using Python. You should generate all 5 reports and include them in your submission.
# 

# In[30]:


import os
import datetime

def validate_clock(filename):
    '''
    Validate if the clock is accurate and report the misalignment in minutes.
    
    Input: filename (str): the name of an image file (a picture of a clock face).
    
    Output: misalignment[ = check_alignment(angle_hour, angle_minute)] (int): the misalignment in minutes.
    '''
    
    clock_RGB = plt.imread(filename)                             # Get the RGB array
    hour_coords, minute_coords = get_clock_hands(clock_RGB)      # Find the coordinates of the hand and minute hands by the RGB array
    angle_hour = get_angle(hour_coords)                          # Get the angles of the hand and minute hands by their coordinates
    angle_minute = get_angle(minute_coords)
    
    return check_alignment(angle_hour, angle_minute)             # Calculate the misalignment between the hour and minute hand

def validate_batch(path, tolerance):
    '''
    Generate quality control report for a batch of clocks with a given folder path, with a pass tolerance.
    
    Input: path (str): a string that indicate the path of a folder containing a batch of clock pictures.
           tolerance (int): a positive integer representing the maximum tolerable number of minutes of misalignment
                            for a clock to pass the quality control check.
    
    Output: batch_X_QC.txt(.txt file): reports some information of the quality control check.
    '''
    
    # Get a list of the file names of clock images in a batch
    dirs = os.listdir(path)
    
    # Find the misalignment for each clock and count the total and pass numbers
    tot_count = 0
    pass_count = 0
    clock_dic_readj = {}         # Store the clock that should be sent back for realignment and its misalignment in a dictionary
    for file in dirs:
        tot_count += 1
        filename = path + '/' + file
        clock_misali = validate_clock(filename)         # Calculate the misalignment
        if clock_misali <= tolerance:                   # Perform a pass judgment by the maximum tolerance
            pass_count += 1
        else:
            clock = file[:-4]                           # Remove '.png' in the filename(obtain the exact clock)
            clock_dic_readj[clock] = clock_misali
    
    # Some calculation for the final report
    # Get the check time in the format YYYY-MM-DD, hh:mm:ss
    checktime = datetime.datetime.now()                 
    checktime_str = checktime.strftime("%Y-%m-%d, %H:%M:%S")
    # Round the batch quality to 1 decimal place, add % later in the report
    quality = round(100 * pass_count / tot_count,1)  
    # List all clocks that need a realignment in decreasing order of misalignment and generate part of the report
    clock_dic_readj = {key: val for key, val in sorted(clock_dic_readj.items(), key=lambda x: x[1], reverse=True)}  
    clock_dic_readj_list = list(clock_dic_readj.items())
    clock_readj_report = ''
    for i in range(len(clock_dic_readj)):
        clock_number = clock_dic_readj_list[i][0]
        misali_time = clock_dic_readj_list[i][1]
        clock_readj_report += (f'%-10s {misali_time}min'%f'{clock_number}' + '\n')    # Align each line vertically   
        
    # Get every line of the report
    report_1 = f'Batch number: {path[-1]}'
    report_2 = 'Checked on '+ checktime_str
    report_3 = f'Total number of clocks: {tot_count}'
    report_4 = f'Number of clocks passing quality control ({tolerance}-minute tolerance): {pass_count}'
    report_5 = f'Batch quality: {quality}%'
    report_6 = 'Clocks to send back for readjustment:'
    report_7 = clock_readj_report
    
    # Generate a report and write it in a .txt file
    with open(f'QC_reports/batch_{path[-1]}_QC.txt', 'w') as mynewreport:   # Replace X by the batch number
        mynewreport.write(report_1 + '\n')
        mynewreport.write(report_2 + '\n')
        mynewreport.write('\n')
        mynewreport.write(report_3 + '\n')
        mynewreport.write(report_4 + '\n')
        mynewreport.write(report_5 + '\n')
        mynewreport.write('\n')
        mynewreport.write(report_6 + '\n')
        mynewreport.write(report_7)

        
# Create a folder called QC_reports
os.makedirs('QC_reports') 
# The tolerance is 3 minutes
tolerance = 3  
# Generate quality control reports for the 5 batches of clocks one by one
for i in range(5):
    path = 'clock_images/batch_'+f'{i}'
    validate_batch(path, tolerance)

# test
# check a single image of clock
# print(validate_clock('clock_images/batch_0/clock_0.png'))
# check a batch of clock pictures
# validate_batch('clock_images/batch_0', 2)
# help(validate_clock)
# help(validate_batch)


# In[ ]:





# In[ ]:




