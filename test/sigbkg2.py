import numpy as np
import pylab as P
from nn_mlp import BackPropogationNetwork,npA 

def bins(xbins,xmin,xmax):
  x=[]
  width=(xmax-xmin)/xbins
  for i in range(xbins+1):
    x.append(i*width+xmin)
  return tuple(x)

def hist1dDraw(v1,bins):
  P.hist(v1,bins,histtype='step')
  #P.hist(vv, bins, normed=1, histtype='step', cumulative=True)
  P.plot()
  P.show()

if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    lFuncs = [None, BackPropogationNetwork.sgm, BackPropogationNetwork.linear]

    bpn = BackPropogationNetwork( (2,4,1), lFuncs, 0.4)

    samples = np.zeros(50, dtype=[('x',  float, 1),('x2',  float, 1), ('y', float, 1)])
    samples['x'] = np.random.normal(40., 5., 50)
    samples['x2'] = np.random.normal(40., 5., 50)
    samples['y'] = np.ones(50)

    samples2 = np.zeros(50, dtype=[('x',  float, 1),('x2',  float, 1), ('y', float, 1)])
    samples2['x'] = np.random.normal(40., 50., 50)
    samples2['x2'] = np.random.normal(40., 50., 50)
    samples2['y'] = np.zeros(50)

    #samples['x'] += generate1dvalues(50,10,20)
    #samples['y'] += np.zeros(50)


    input   = np.array([ [ll[0],ll[1]] for ll in zip(samples['x'],samples['x2']) ])
    input2  = np.array([ [ll[0],ll[1]] for ll in zip(samples2['x'],samples2['x2']) ])
    target = np.array([samples['y']]).T
    target2 = np.array([samples2['y']]).T

    max = 100000
    #lnErr = 1e-5
    #lnErr = 0.2
    lnErr = 5.0

    for i in range(max+1):
        err = 0.0;
        for ii in range(len(input)):
            err += bpn.trainEpoch(npA(input[ii]),npA(target[ii]))
            err += bpn.trainEpoch(npA(input2[ii]),npA(target2[ii]))
        if i%2500 == 0:
        #if i%20 == 0:
            print "iteration {0}\tError : {1:0.6f}".format(i,err)
            #bpn.NN_cout("Train_"+str(i))
        if err <= lnErr:
            print "Min error reached at {0}".format(i)
            break

    #bpn.setWeights()

    bpn.printWeights()
    for i in range(len(input)):
        output = bpn.run(npA(input[i]))
        print "in :"+str(input[i])+", ou : "+str(output)

    for i in range(len(input2)):
        output = bpn.run(npA(input2[i]))
        print "in2 :"+str(input2[i])+", ou2 : "+str(output)


    #bins = bins(20,0,100)
    #P.hist(v1,bins,histtype='step')
    #P.hist(v1,bins,histtype='step')


    """
    plt.figure(figsize=(10,5))
    # Draw real function
    x,y = samples['x'],samples['y']
    plt.plot(x,y,color='b',lw=1)
    # Draw network approximated function
    for i in range(len(input)):
        y[i] = bpn.run(npA(input[i]))
    for i in range(len(input)):
        y[i] = bpn.run(npA(input[i]))
    plt.plot(x,y,color='r',lw=3)
    plt.axis([0,1,0,1])
    plt.show()
    """
