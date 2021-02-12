''' ------------------------------------PERCEPTRON VISUALIZATION------------------------------------------------ '''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import random
import math

'''select the dimension of data point (preferred: 1-dimensional or 2-dimensional)'''
dimension_of_data_point=2

if(dimension_of_data_point==1):
      '''
      Perceptron works on linearly separable data. So, our data points will be in 2 categories (-1 and +1).
      For linear separable datapoints, I draw a line and then choose  data points randomly and then put all data points
      lying on one side of line in +1 category and on other side in -1 category
      Equation of line passing through origin = ax+by=0 . So, I am choosing here a and b.
      '''
      a=random.randint(-10,10)
      b=random.randint(-10,10)

      '''no of data points'''
      no_points=random.randint(2,20)

      '''data_points is list containing the data points in the form [x,y,category] where x component of all
      data points is 1(data point in y dimension) and category is +1 or -1.'''
      data_points=[]

      '''loop to select the data points randomly.'''
      for point in range(no_points):
            y=random.randint(-10,10)
            val=a*1+b*y
            if(val>=0):
                  data_points.append([1,y,1])
            else:
                  data_points.append([1,y,-1])

      '''positive_x will contains the x-coordinates of data points with category +1
      positive_y will contains the y-coordinates of data points  with category +1
      negative_x will contains the x-coordinates of data points with category -1
      negative_y will contains the y-coordinates of data points with category -1'''
      positive_x=[m[0] for m in data_points if m[2]==1]
      positive_y=[m[1] for m in data_points if m[2]==1]
      negative_x=[m[0] for m in data_points if m[2]==-1]
      negative_y=[m[1] for m in data_points if m[2]==-1]

      ''' Condition check if-else loop , if both category has atleast one data point ,only then apply perceptron'''
      if(len(positive_x)==0):
            print("All points are of -1 category. So, No point of applying perceptron as only one category.")
      elif(len(negative_x)==0):
            print("All points are of +1 category. So, No point of applying perceptron as only one category.")
      else:
          '''calculating range of x-axis and y-axis values'''
          x_max=max(positive_x+negative_x)
          x_min=min(positive_x+negative_x)
          y_max=max(positive_y+negative_y)
          y_min=min(positive_y+negative_y)

          ''' Classifier is a list that will contain the w vector corresponding to w^T x =0 . The line perpendicular 
          to w will be the line separating the two categories datapoints.'''
          classifier=[]

          '''Initializing w as vector [1,1] .We can initialize with any value '''
          w=[1,1]
          classifier.append([w[0],w[1]])

          '''Run the loop till convergences i.e. The line perpendicular to updated w  will be able to categories all 
          +1 points on one side  and all -1 points on the other side'''
          while(True):
                error=0
                for i in range(0,len(data_points)):
                    m=data_points[i]
                    if (w[0]*m[0]+w[1]*m[1]<0 and m[2]==1):
                        error=1
                        w[0]=w[0]+m[0]
                        w[1]=w[1]+m[1]
                        classifier.append([w[0],w[1]])
                              
                    elif (w[0]*m[0]+w[1]*m[1]>=0 and m[2]==-1):
                        error=1
                        w[0]=w[0]-m[0]
                        w[1]=w[1]-m[1]
                        classifier.append([w[0],w[1]])

                if(error==1):
                    continue
                else:
                    break

          '''Rest of the portion of this if loop is just for plotting purpose'''
          total=len(classifier)+1
          rows=math.ceil(math.sqrt(total))
          fig = plt.figure(figsize=(rows*3,rows*3))
          plt.title("Classification after successive iterations of Perceptron\n\n")
          plt.axis('off')
          ax=fig.add_subplot(rows,rows,1)
          ax.title.set_text("data points")
          
          sns.scatterplot(x=positive_x,y=positive_y,color="red",label='+1')
          sns.scatterplot(x=negative_x,y=negative_y,color="green",label='-1')
          ax.axes.xaxis.set_ticklabels([])
          ax.axes.yaxis.set_ticklabels([])
          ax.legend(loc='upper left')
          plt.xlim(x_min-3,x_max+3)
          plt.ylim(y_min-3,y_max+3)
          item=2
              
          for j in range(0,len(classifier)):
                m=classifier[j]
                ax=fig.add_subplot(rows,rows,item)
                sns.scatterplot(x=positive_x,y=positive_y,color="red",label='+1')
                sns.scatterplot(x=negative_x,y=negative_y,color="green",label='-1')

                if(m[1]!=0):
                    xx=[x_min-2,0,x_max+2]
                    yy=[(-m[0]*p)/m[1] for p in xx]
                else:
                    xx=[0,0,0]
                    yy=[y_min-2,0,y_max+2]
                
                plt.plot(xx,yy,color="blue",linewidth=0.5)
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                plt.xlim(x_min-3,x_max+3)
                plt.ylim(y_min-3,y_max+3)
                ax.title.set_text("Iteration number : "+str(j+1))
                ax.legend(loc='upper left')
                item+=1

          plt.show()


elif (dimension_of_data_point==2):
      '''
      Perceptron works on linearly separable data. So, our data points will be in 2 categories (-1 and +1).
      For linear separable datapoints, I draw a plane and then choose  data points randomly and then put all data points
      lying on one side of plane in +1 category and on other side in -1 category
      Equation of plane passing through origin is ax+by+cz0 . So, I am choosing here a, b and c.
      '''
      a=random.randint(-10,10)
      b=random.randint(-10,10)
      c=random.randint(-10,10)

      '''no of data points'''
      no_points=random.randint(2,20)

      '''data_points is list containing the data points in the form [x,y,z,category] where x component of all
      data points is 1(data point in y and z dimension) and category is +1 or -1.'''
      data_points=[]

      '''loop to select the data points randomly.'''
      for point in range(no_points):
          y=random.randint(-10,10)
          z=random.randint(-10,10)
          val=a*1+b*y+c*z
          if(val>=0):
              data_points.append([1,y,z,1])
          else:
              data_points.append([1,y,z,-1])

      '''positive_x will contains the x-coordinates of data points with category +1
      positive_y will contains the y-coordinates of data points  with category +1
      positive_z will contains the z-coordinates of data points  with category +1
      negative_x will contains the x-coordinates of data points with category -1
      negative_y will contains the y-coordinates of data points with category -1
      negative_z will contains the z-coordinates of data points with category -1'''

      positive_x=[m[0] for m in data_points if m[3]==1]
      positive_y=[m[1] for m in data_points if m[3]==1]
      positive_z=[m[2] for m in data_points if m[3]==1]
      negative_x=[m[0] for m in data_points if m[3]==-1]
      negative_y=[m[1] for m in data_points if m[3]==-1]
      negative_z=[m[2] for m in data_points if m[3]==-1]

      ''' Condition check if-else loop , if both category has atleast one data point ,only then apply perceptron'''
      if(len(positive_x)==0):
            print("All points are of -1 category. So, No point of applying perceptron as only one category.")
      elif(len(negative_x)==0):
            print("All points are of +1 category. So, No point of applying perceptron as only one category.")
      else:
          '''calculating range of x-axis, y-axis and z_axis values'''
          x_max=max(positive_x+negative_x)
          x_min=min(positive_x+negative_x)
          y_max=max(positive_y+negative_y)
          y_min=min(positive_y+negative_y)
          z_max=max(positive_z+negative_z)
          z_min=min(positive_z+negative_z)

          ''' Classifier is a list that will contain the w vector corresponding to w^T x =0 . The plane  perpendicular 
          to w will be the plane separating the two categories datapoints.'''
          classifier=[]

          '''Initializing w as vector [1,1] .We can initialize with any value '''
          w=[1,1,1]
          classifier.append([w[0],w[1],w[2]])

          '''Run the loop till convergences i.e. The plane perpendicular to updated w  will be able to categories all 
          +1 points on one side  and all -1 points on the other side'''
          while(True):
              error=0
              for i in range(0,len(data_points)):
                  m=data_points[i]
                  if(w[0]*m[0]+w[1]*m[1]+w[2]*m[2]<0 and m[3]==1):
                      error=1
                      w[0]=w[0]+m[0]
                      w[1]=w[1]+m[1]
                      w[2]=w[2]+m[2]
                      classifier.append([w[0],w[1],w[2]])
                            
                  elif (w[0]*m[0]+w[1]*m[1]+w[2]*m[2]>=0 and m[3]==-1):
                      error=1
                      w[0]=w[0]-m[0]
                      w[1]=w[1]-m[1]
                      w[2]=w[2]-m[2]
                      classifier.append([w[0],w[1],w[2]])

              if(error==1):
                  continue
              else:
                  break

          '''Rest of the portion of this else loop is just for plotting purpose'''
          total=len(classifier)+1
          rows=math.ceil(math.sqrt(total))
          fig = plt.figure(figsize=(rows*3,rows*3))
          plt.title("Classification after successive iterations of Perceptron\n\n(Red color - category +1  and Green color - category -1\n\n")
          plt.axis('off')
          ax=fig.add_subplot(rows,rows,1, projection='3d')
          for i in range(len(positive_x)):
              ax.scatter(positive_x[i],positive_y[i],positive_z[i],color="red")
          for i in range(len(negative_x)):
              ax.scatter(negative_x[i],negative_y[i],negative_z[i],color="green")
          ax.axes.xaxis.set_ticklabels([])
          ax.axes.yaxis.set_ticklabels([])
          ax.axes.zaxis.set_ticklabels([])
          ax.set_xlim(x_min-3,x_max+3)
          ax.set_ylim(y_min-3,y_max+3)
          ax.set_zlim(z_min-3,z_max+3)
          ax.title.set_text("Data Points ")
          item=2
              
          for j in range(0,len(classifier)):
              m=classifier[j]
              ax=fig.add_subplot(rows,rows,item, projection='3d')
              for i in range(len(positive_x)):
                  ax.scatter(positive_x[i],positive_y[i],positive_z[i],color="red")
              for i in range(len(negative_x)):
                  ax.scatter(negative_x[i],negative_y[i],negative_z[i],color="green")

              if(m[2]!=0):
                  xx = np.linspace(x_min-3,x_max+3,10)
                  yy = np.linspace(y_min-3,y_max+3,10)
                  XX,YY = np.meshgrid(xx,yy)
                  zz=(-m[0]*XX-m[1]*YY)/m[2]
              else:
                  xx=np.linspace(x_min-3,x_max+3,10)
                  zz=np.linspace(z_min-3,z_max+3,10)
                  XX,zz=np.meshgrid(xx,zz)
                  YY=(-m[0]*XX)/m[1]  
              
              ax.plot_surface(XX,YY,zz,alpha=0.5)
              ax.axes.xaxis.set_ticklabels([])
              ax.axes.yaxis.set_ticklabels([])
              ax.axes.zaxis.set_ticklabels([])
              ax.title.set_text("Iteration no : "+str(j+1))
              item+=1

          plt.show()
          
else:
  print("please keep the dimension of data point less than 3 . Otherwise it will not be easy to visualise. Try again!")
