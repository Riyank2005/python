#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# numpy
import numpy as np
a=np.array(20)
print('dimension:',a.ndim)
print('shape:',a.shape)
print(a)


# In[ ]:


import numpy as np
a=np.array([2,9,6,5,'a'])
print('dimension:',a.ndim)
print('shape:',a.shape)
print(a)


# In[ ]:


import numpy as np
a=np.array([2,9,6,5,15])
print('dimension:',a.ndim)
print('shape:',a.shape)
print(a)


# In[ ]:


import numpy as np
a=np.array([[2,3,4,5],[6,7,8,9],[1,6,5,3]])
print('dimension:',a.ndim)
print('shape:',a.shape)
print(a)


# In[ ]:


import numpy as np
a=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]])
print('dimension:',a.ndim)
print('shape:',a.shape)
# print(a)
# a=a.reshape(2,1,3,3)
# a=a.reshape(1,2,3,3)

# a=a.reshape(-1)
a=a.reshape(2,3,3)



print(a)


# In[47]:


import numpy as np
l=[]
for i in range(1,211):
    l.append(i)
a=np.array(l)
print('dimension:',a.ndim)
print('shape:',a.shape)
a=a.reshape(5,6,7)
print(a[3,1,6])
print(a[0:3:2,::2,3:6])
print(a[0:,4:6,3])
print(a[])

print(a)


# In[ ]:


s=0
for i in arr:
    for j in i:
        st=j[1]


# In[3]:


import numpy as np
l=[]
for i in range(18):
    l.append(i+1)
sum=0
a=np.array(l)
a=a.reshape(3,2,3)
for i in a:
    for j in i:
        sum+=j[1]
print(sum)


# In[5]:


import numpy as np
a1=np.array([1,2,3])
a2=np.array([4,5,6,7])
print(a1.shape)
print(a2.shape)
c=np.concatenate((a1,a2))
print(c.shape)
print(c)


# In[20]:


# import numpy as np
# a1=np.array([[[1,2,3],[4,5,6]],[7,8,9],[[10,11,12]]])
# a2=np.array([[[13,14,15],[16,17,18]],[[19,20,21],[22,23,24]]])
# print(a1.shape)
# print(a2.shape)
# c=np.concatenate((a1,a2),axis=1)
# print(c.shape)
# print(c)


# In[19]:


# import numpy as np
# a1 = np.array([[[1, 2, 3], [4, 5, 6]], [7, 8, 9], [[10, 11, 12]]])
# a2 = np.array([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]])
# print(a1.shape)
# print(a2.shape)
# c = np.concatenate((a1, a2), axis=1)
# print(c.shape)
# print(c)


# In[25]:


import matplotlib.pyplot as plt
import numpy as np
# x=np.array([2,4,5,3,9,6])
y=np.array([3,6,4,3,2,7])
plt.plot(x,y,marker='o',ms=10,mec='black',mfc='red',ls='dashdot',lw=2.5,c='black')
# ^ * s o


plt.show()


# In[12]:


import matplotlib as m
help(m.markers)


# In[26]:


import matplotlib.pyplot as plt
import numpy as np
# x=np.array([2,4,5,3,9,6])
y=np.array([3,6,4,3,2,7])
plt.plot(y,marker='o',ms=10,mec='black',mfc='red',ls='dashdot',lw=2.5,c='black')
# ^ * s o
plt.show()


# In[63]:


import matplotlib.pyplot as plt
import numpy as np

y1=np.array([2,4,5,3,9,6])
y2=np.array([3,6,4,3,2,7])

plt.plot(y1,marker='o',ms=10,mec='black',mfc='red',ls='dashdot',lw=2.5,c='black',label='profit')
plt.plot(y2,marker='o',ms=10,mec='black',mfc='red',ls='dotted',lw=2.5,c='black',label='revenue')
plt.legend(loc=0,title='abc')
d={'color':'r','size':15,'family':'serif'}
plt.xlabel('x value',fontdict=d)
plt.ylabel('y value',fontdict=d)
plt.title('data',fontdict=d,loc='center')
plt.grid(axis='x',ls='--')




# ^ * s o


plt.show()


# In[129]:


import matplotlib.pyplot as plt
import numpy as np

y1=np.array([2,4,5,3,9,6])
y2=np.array([3,6,4,3,2,7])

# plt.plot(y1,y2,'s:b')
plt.plot(y1,y2,'-.,r')

plt.show()


# In[86]:


import matplotlib.pyplot as plt
import numpy as np

y1=np.array([2,4,5,3,9,6])
y2=np.array([3,6,4,3,2,7])
plt.subplot(2,1,1)
plt.plot(y1)
plt.title('data1')
plt.subplot(2,1,2)
plt.plot(y2)
plt.title('data2')



plt.show()


# In[103]:


import matplotlib.pyplot as plt
import numpy as np

y1=np.array([2,4,5,3,9,6])
y2=np.array([3,6,4,3,2,7])
y3=np.array([3,6,4,3,2,7])
y4=np.array([3,6,4,3,2,7])
y2=np.array([3,6,4,3,2,7])

plt.subplot(2,3,1)
plt.plot(y1)
plt.title('data1',color='m')
plt.subplot(2,3,2)
plt.plot(y2)
plt.title('data2',color='y')
plt.subplot(2,3,3)
plt.title('data3',color='m')
plt.plot(y3)
plt.title('data1',color='y')
plt.subplot(2,3,4)
plt.plot(y4)
plt.title('data1',color='m')
plt.subplot(2,3,5)
plt.plot(y1)
plt.title('data1',color='y')
plt.subplot(2,3,6)
plt.plot(y2)
plt.title('data1',color='m')



plt.show()


# In[104]:


import matplotlib.pyplot as plt
import numpy as np

y1=np.array([2,4,5,3,9,6])
y2=np.array([3,6,4,3,2,7])

plt.plot(y1,marker='o',ms=10,mec='black',mfc='red',ls='dashdot',lw=2.5,c='black',label='profit')
plt.plot(y2,marker='o',ms=10,mec='black',mfc='red',ls='dotted',lw=2.5,c='black',label='revenue')
plt.legend(loc=0,title='abc')
d={'color':'r','size':15,'family':'serif'}
plt.xlabel('x value',fontdict=d)
plt.ylabel('y value',fontdict=d)
plt.title('data',fontdict=d,loc='center')
plt.grid(axis='x',ls='--')




# ^ * s o


plt.show()


# In[126]:


import matplotlib.pyplot as plt
import numpy as np
y=[35,25,15,25]
data=['math','sci','py','java']
mycl=['r','y','g','c']
e=[0.8,0.2,0.4,0.5]
plt.pie(y,startangle=180,labels=data,colors=mycl,shadow=True,explode=e,autopct='%.1f%%')

plt.show()


# In[139]:


import matplotlib.pyplot as plt
import numpy as np
x=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
y=np.array([211000,183300,224700,222700,209600,201400,295500,361400,234000,266700,412800,300200])
plt.plot(x,y,marker='o',ms=5,mec='black',mfc='blue',ls='dotted',lw=3,c='r',label='profit')
# plt.plot(y2,marker='o',ms=10,mec='black',mfc='red',ls='dotted',lw=2.5,c='black',label='revenue')
plt.legend(loc='lower right',title='abc')
plt.xlabel(' Month Number')
plt.ylabel('Total Profits')
plt.title('data',fontdict=d,loc='center')
plt.grid(axis='x',ls='--')



# In[146]:


import matplotlib.pyplot as plt
import numpy as np
y=[1.35,3.60,2.25,1.8]
data=['Engineering','manufacturing ','sales','profit']
mycl=['r','y','g','c']
plt.pie(y,startangle=0,labels=data,shadow=True,autopct='%2.f%%')

plt.show()


# In[40]:


# Subplot 3 (Bar chart for daily workout durations):
# Use a blue color for bars.
# Title: "Daily Workout Durations (Bar Chart)"
# Subplot 4 (Histogram for workout durations distribution):
# Use a skyblue color for bars. Take value of bins as 10
# Title: "Workout Durations Distribution (Histogram)"
# Subplot 5 (Pie chart for calories burned distribution over the weeks):
# startangle should be 90 degree and also display percentage in pie chart with 1 digits after one decimal place.
# Also display labels
# Title: "Calories Burned Distribution (Pie Chart)"
# Subplot 6 (Line plot for the rate of change in workout durations):
# Use a red line with triangular markers (marker size 8).
# Title: "Rate of Change in Workout Durations"
# Features and Customizations:
# Utilize different colors for each plot.
# Add markers to the line plots for emphasis.
# Adjust marker sizes for better visibility.
# Formula and Calculation:
# The rate of change in workout durations can be calculated using the formula:
# Let's consider an example to illustrate this. Suppose we have the following workout duration data for four consecutive 
# days:
# Workout Duration already given in list above at top (workout_durations_data):
# Day 1: Workout Duration = 40 minutes
# Day 2: Workout Duration = 50 minutes
# Day 3: Workout Duration = 45 minutes
# Day 4: Workout Duration = 55 minutes
# We want to calculate the rate of change in workout durations for each day.
# The negative rate of change between Day 2 and Day 3 indicates a decrease in workout duration, while the positive rate of 
# change between Day 3 and Day 4 indicates an increase.
# In the context of the fitness data visualization, the rate of change plot will show how the workout durations are changing 
# on a daily basis, providing insights into trends and fluctuations in the individual's exercise routine.
# Instructions:
# Implement the required calculations using nested loops.
# Ensure that each subplot is labeled appropriately.
# Customize the visualizations with specific colors, markers, and marker sizes.
# Display the main title for the entire figure.
import matplotlib.pyplot as plt
import numpy as np


workout_durations_data = [40, 50, 45, 55, 60, 30, 40, 50, 45, 55, 60, 30, 40, 50, 45, 55, 60, 30, 40, 50, 45, 55, 60, 30, 40, 50, 45, 55]
calories_burned_data= [200, 250, 220, 270, 300, 150, 200, 250, 220, 270, 300, 150, 200, 250, 220, 270, 300, 150, 200, 250, 220, 270, 300, 150, 200, 250, 220, 270]
week_dur=[]
week_cb=[]
for i in range (0,len(workout_durations_data)//7):
    sd,scb=0,0
    for j in range((i*7),((i+1)*7)):
        sd+=workout_durations_data[j]
        scb+=calories_burned_data[j]
    week_dur.append(sd/7)
    week_cb.append(scb/7)   
print(week_dur)
print(week_cb)
days=list(range(1,29))
# subploat1
plt.subplot(3,2,1)
plt.plot(days,workout_durations_data,marker='o',c='b',ms=8 )
plt.title( "Daily Workout Durations")
# subploat2
plt.subplot(3,2,2)
plt.plot([1,2,3,4],week_dur,marker='o',ms=8,c='green')
plt.plot([1,2,3,4],week_cb,marker='o',c='orange',ms=8 )
plt.title( "Weekly Averages")
# subploat3
plt.subplot(3, 2, 3)
plt.bar(days, workout_durations_data, color='blue')
plt.title("Daily Workout Durations")
plt.xlabel("Day")
plt.ylabel("Duration")
# subploat4
plt.subplot(3, 2, 4)
plt.hist(workout_durations_data, color='skyblue',bins=10)
plt.title("Workout Durations Distribution (Histogram)")
# subploat5
plt.subplot(3, 2, 5)
plt.pie(week_cb,startangle=90,shadow=True,autopct='%1.f%%')
rate_of_change = np.diff(workout_durations_data)
days_for_rate_of_change = days[1:]
plt.subplot(3, 2, 6)
plt.plot(days_for_rate_of_change, rate_of_change, marker='^', color='red', markersize=8)
plt.title("Rate of Change in Workout Durations")
plt.xlabel("Day")
plt.ylabel("Rate of Change (minutes)")

plt.show()





# In[69]:


import matplotlib.pyplot as plt
import numpy as np
scores= [[31, 12, 19, 53], 
 [67, 48, 95, 83], 
 [59, 67, 13, 59], 
 [62, 29, 99, 88], 
 [87, 91, 69, 76]]
a=np.array(scores)
m1=np.max(a[:,2])
print(m1)
m2=np.min(a[2,:])
print(m2)
sum_scores = np.sum(a, axis=1).reshape(5,1)
a = np.concatenate((a, sum_scores), axis=1)
print(a)
plt.pie(sum_scores.flatten(), autopct='%.f%%', startangle=90)
plt.title("Total Runs (Sum of Scores) for Each Batsman Across 4 Matches")
# m3=a[1,:]
# print(m3)
# plt.bar(m3,[1,2,3,4,5], color='blue')
plt.subploat(4,5,1)
plt.bar(range(1, len(m3) + 1), m3, color='blue')  
plt.xlabel('Match Number')
plt.ylabel('Runs')
plt.title('Runs by Batsman in Each Match')
plt.show()

plt.show()









# In[71]:


import matplotlib.pyplot as plt
import numpy as np

scores = [[31, 12, 19, 53], 
          [67, 48, 95, 83], 
          [59, 67, 13, 59], 
          [62, 29, 99, 88], 
          [87, 91, 69, 76]]

a = np.array(scores)
m1 = np.max(a[:, 2])
print(m1)

m2 = np.min(a[2, :])
print(m2)

sum_scores = np.sum(a, axis=1).reshape(5, 1)
a = np.concatenate((a, sum_scores), axis=1)
print(a)
plt.subploat(1,2,1)

plt.pie(sum_scores.flatten(), autopct='%.f%%', startangle=90)
plt.title("Total Runs (Sum of Scores) for Each Batsman Across 4 Matches")

m3 = a[1, :]
plt.subplot(1, 2, 2)
plt.bar(range(1, len(m3) + 1), m3, color='blue')
plt.xlabel('Match Number')
plt.ylabel('Runs')
plt.title('Runs by Batsman in Each Match')

plt.show()


# In[74]:


import matplotlib.pyplot as plt
import numpy as np

scores = [[31, 12, 19, 53], 
          [67, 48, 95, 83], 
          [59, 67, 13, 59], 
          [62, 29, 99, 88], 
          [87, 91, 69, 76]]

a = np.array(scores)
m1 = np.max(a[:, 2])
print(m1)

m2 = np.min(a[2, :])
print(m2)

sum_scores = np.sum(a, axis=1).reshape(5, 1)
a = np.concatenate((a, sum_scores), axis=1)
print(a)

plt.subplot(1, 2, 1)
plt.pie(sum_scores.flatten(), autopct='%.f%%', startangle=90)
plt.title("Total Runs (Sum of Scores) for Each Batsman Across 4 Matches")

m3 = a[1],[:4]
plt.subplot(1, 2, 2)
plt.bar(range(1, len(m3) + 1), m3, color='blue')
plt.xlabel('Match Number')
plt.ylabel('Runs')
plt.title('Runs by Batsman in Each Match')

plt.show()


# In[75]:


import matplotlib.pyplot as plt
import numpy as np

scores = [[31, 12, 19, 53], 
          [67, 48, 95, 83], 
          [59, 67, 13, 59], 
          [62, 29, 99, 88], 
          [87, 91, 69, 76]]

a = np.array(scores)
m1 = np.max(a[:, 2])
print(m1)

m2 = np.min(a[2, :])
print(m2)

sum_scores = np.sum(a, axis=1).reshape(5, 1)
a = np.concatenate((a, sum_scores), axis=1)
print(a)

plt.subplot(1, 2, 1)
plt.pie(sum_scores.flatten(), autopct='%.f%%', startangle=90)
plt.title("Total Runs (Sum of Scores) for Each Batsman Across 4 Matches")

m3 = a[1, :4]  # Corrected: Select the second row (batsman 2) for the first 4 matches
plt.subplot(1, 2, 2)
plt.bar(range(1, len(m3) + 1), m3, color='blue')
plt.xlabel('Match Number')
plt.ylabel('Runs')
plt.title('Runs by Batsman in Each Match')

plt.show()


# In[ ]:




