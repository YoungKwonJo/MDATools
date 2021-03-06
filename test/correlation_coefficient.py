
from math import *

"""

ref. https://en.wikipedia.org/wiki/Correlation_and_dependence#Pearson.27s_product-moment_coefficient

def correlation coefficient:
  data = [{x1,y1},{x2,y2},{x3,y3}]
  meanX = sum_{i=0} {x_i}/len(data)
  meanY = sum_{i=0} {y_i}/len(data)

  return r_{xy} = { sum_{i=0} { (x_i -meanX)*(y_i -meanY ) } }/{ sqrt{sum_{i=0} { (x_i -meanX )**2 } } * sqrt{sum_{i=0} { (y_i -meanY)**2 } } }

"""

def round2(x,i):
  if(i==0) : return round(x) 
  else     : return round(x*(10*i))/(10*i)

def mean(x):
  xsum=sum(x)
  if len(x)==0 : return False
  else         : return (xsum/len(x))

def correlation_coefficient(data):
  x =[]
  y =[]
  for i in data:
    x.append( i[0] )
    y.append( i[1] )
  return correlation_coefficientA2(x,y)

def correlation_coefficientA2(x,y):
  meanX = mean(x)
  meanY = mean(y)
  return correlation_coefficientA4(x,y,meanX,meanY)

def correlation_coefficientA4(x,y,meanX,meanY):
  sumXY=0.
  sumX =0.
  sumY =0.
  for i,xx in enumerate(x):
    x_=(x[i]-meanX)
    y_=(y[i]-meanY)
    sumXY +=x_*y_
    sumX  +=x_*x_
    sumY  +=y_*y_

  return [sumXY/(sqrt(sumX)*sqrt(sumY)),meanX,meanY]

#data = [ [1,2], [2,3], [5,2], [3,7], [5,2], [3,2] ]
data =[]
for i in range(100):
  data.append([i,i*i])

r = correlation_coefficient(data)
print str(data)
print "r = "+str(round(r[0]*100))+"%, meanX="+str(round2(r[1],2))+", meanY="+str(round2(r[2],2))







