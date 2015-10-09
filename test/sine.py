import numpy as np
from nn_mlp import BackPropogationNetwork,npA 

if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    print "Learning the sin function"
    lFuncs = [None, BackPropogationNetwork.sgm, BackPropogationNetwork.linear]

    bpn = BackPropogationNetwork( (1,4,1), lFuncs, 0.4)

    samples = np.zeros(20, dtype=[('x',  float, 1), ('y', float, 1)])
    samples['x'] = np.linspace(0,1,20)
    samples['y'] = np.sin(samples['x']*np.pi)

    input  = np.array([samples['x']]).T
    target = np.array([samples['y']]).T
    max = 100000
    #lnErr = 1e-5
    lnErr = 1e-2

    for i in range(max+1):
        err = 0.0;
        for ii in range(len(input)):
            err += bpn.trainEpoch(npA(input[ii]),npA(target[ii]))
        if i%2500 == 0:
        #if i%20 == 0:
            print "iteration {0}\tError : {1:0.6f}".format(i,err)
            #bpn.NN_cout("Train_"+str(i))
        if err <= lnErr:
            print "Min error reached at {0}".format(i)
            break

    #bpn.setWeights()

    bpn.printWeights()

    plt.figure(figsize=(10,5))
    # Draw real function
    x,y = samples['x'],samples['y']
    plt.plot(x,y,color='b',lw=1)
    # Draw network approximated function
    for i in range(len(input)):
        y[i] = bpn.run(npA(input[i]))
    plt.plot(x,y,color='r',lw=3)
    plt.axis([0,1,0,1])
    plt.show()

