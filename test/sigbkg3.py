import numpy as np
import pylab as P
from nn_mlp import BackPropogationNetwork,npA 
import random

def bins(xbins,xmin,xmax):
  x=[]
  width=(xmax-xmin)/xbins
  for i in range(xbins+1):
    x.append(i*width+xmin)
  return tuple(x)

def circle(x,y,r, events=50):
  a,b=[],[]
  for i in range(events):
    t = i*2*np.pi/events
    x1 = x+r*np.cos(t)*(0.8+0.2*random.random())
    y1 = y+r*np.sin(t)*(0.8+0.2*random.random())
    a.append(x1)
    b.append(y1)
  return a,b

def hist1dDraw(v1,bins):
  P.hist(v1,bins,histtype='step')
  #P.hist(vv, bins, normed=1, histtype='step', cumulative=True)
  P.plot()
  P.show()

if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    lFuncs = [None, BackPropogationNetwork.sgm, BackPropogationNetwork.linear]

    bpn = BackPropogationNetwork( (2,3,1), lFuncs, 0.2)

    sam = 50
    samples = np.zeros(sam, dtype=[('x',  float, 1),('x2',  float, 1), ('y', float, 1)])
    a1,b1 = circle(40,40,10,sam)
    samples['x']  = np.array([a1])
    samples['x2'] = np.array([b1])
    samples['y'] = np.ones(sam)

    sam2 =50
    samples2 = np.zeros(sam2, dtype=[('x',  float, 1),('x2',  float, 1), ('y', float, 1)])
    a2,b2 = circle(40,40,40,sam2)
    samples2['x']  = np.array([a2]) 
    samples2['x2'] = np.array([b2])
    samples2['y'] = np.zeros(sam2)

    #samples['x'] += generate1dvalues(50,10,20)
    #samples['y'] += np.zeros(50)


    input   = np.array([ [ll[0],ll[1]] for ll in zip(samples['x'],samples['x2']) ])
    input2  = np.array([ [ll[0],ll[1]] for ll in zip(samples2['x'],samples2['x2']) ])
    target = np.array([samples['y']]).T
    target2 = np.array([samples2['y']]).T

    #"""
    #max = 100000
    max = 10000
    #lnErr = 1e-5
    lnErr = 0.2

    for i in range(max+1):
        err = 0.0;
        for ii in range(len(input)):
            err += bpn.trainEpoch(npA(input[ii]),npA(target[ii]))
        #for ii in range(len(input2)):
            err += bpn.trainEpoch(npA(input2[ii]),npA(target2[ii]))
        if i%2500 == 0:
        #if i%20 == 0:
            print "iteration {0}\tError : {1:0.6f}".format(i,err)
            #bpn.NN_cout("Train_"+str(i))
        if err <= lnErr:
            print "Min error reached at {0}".format(i)
            break
    """
    weight =[np.array([[ 1.17370646, -1.02464046,  0.04103199],
       [-0.20393482, -0.05878576,  0.24338135],
       [-0.19635129, -0.04882523,  0.24217662],
       [ 0.19115956,  0.07714156, -0.00614085]]), np.array([[ 0.19762054,  0.1643754 ,  0.1816128 , -0.3159388 ,  0.14438186]])]

    bpn.setWeights(weight)
    """

    bpn.printWeights()
    from ROOT import *
    h2 = TH2F("h2","h2",100,0,100,100,0,100)
    for i in range(len(input)):
        output = bpn.run(npA(input[i]))
        print "in :"+str(input[i])+", ou : "+str(output)
        h2.Fill(input[i][0], input[i][1], output+1.0)

    for i in range(len(input2)):
        output = bpn.run(npA(input2[i]))
        print "in2 :"+str(input2[i])+", ou2 : "+str(output)
        h2.Fill(input2[i][0], input2[i][1], output+1.0)

    gStyle.SetPalette(1)
    h2.SetMinimum(0.0)
    h2.SetMaximum(3.0)
    h2.Draw("colz")


