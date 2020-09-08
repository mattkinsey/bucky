import numpy as np

def IS(x, l, u, alp):
    return (u-l) + 2./alp*(l-x)*(x<l) + 2./alp*(x-u)*(x>u)

def WIS(x, q, x_q):
    #todo sort q and x_q based on q
    K = len(q)//2
    alps = np.array([1-q[-i-1]+q[i] for i in range(K)])
    Fs = np.array([[x_q[i],x_q[-i-1]] for i in range(K)])
    m = x_q[K+1]
    w0 = .5
    wk = alps/2.
    ret = 1./(K+1.)*(w0*2*np.abs(x-m) + np.sum(wk*IS(x,Fs[:,0], Fs[:,1], alps)))
    return ret

