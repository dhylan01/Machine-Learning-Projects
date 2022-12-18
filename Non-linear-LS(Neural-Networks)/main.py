import math
import random

import matplotlib.pyplot
import numpy
import scipy.io
import scipy.io as sio
import numpy as np
from itertools import combinations
from pathlib import Path
from matplotlib import pyplot as plt

#we want a derivitive function given an f(x)
# weights

def init_weights(weight_num,val,uni):
    if uni:
        #numpy.random.seed(1)
        return np.random.uniform(-val,val,(1,weight_num))
        #return numpy.random.rand(1,weight_num)
    w = numpy.ones((1,weight_num))*val
    #w = numpy.random.rand(1,weight_num)
    return w

def deriv( weights, x_vals):
    ret = numpy.zeros((len(weights[0]), 1))
    weights = weights[0]

    scalar = 0
    temp = math.tanh(
        weights[scalar + 1] * x_vals[0] + weights[scalar + 2] * x_vals[1] + weights[scalar + 3] * x_vals[2] + weights[
            scalar + 4])

    ret[0][0] = temp
    temp = math.pow(temp,2)

    ret[1][0] = weights[scalar] * x_vals[0] * (1-temp)
    ret[2][0] = weights[scalar] * x_vals[1] * (1-temp)
    ret[3][0] = weights[scalar] * x_vals[2] * (1-temp)
    ret[4][0] = weights[scalar] * (1-temp)
    scalar = 5

    temp = math.tanh(weights[scalar + 1] * x_vals[0] + weights[scalar + 2] * x_vals[1] + weights[scalar + 3] * x_vals[2] + weights[scalar + 4])
    ret[5][0] = temp

    temp = math.pow(temp,2)

    ret[6][0] = weights[scalar] * x_vals[0] * (1-temp)
    ret[7][0] = weights[scalar] * x_vals[1] * (1-temp)
    ret[8][0] = weights[scalar] * x_vals[2] * (1-temp)
    ret[9][0] = weights[scalar] * (1-temp)
    scalar = 10

    temp = math.tanh(weights[scalar + 1] * x_vals[0] + weights[scalar + 2] * x_vals[1] + weights[scalar + 3] * x_vals[2] + weights[scalar + 4])

    ret[10][0] = temp
    temp = math.pow(temp,2)

    ret[11][0] = weights[scalar] * x_vals[0] * (1-temp)
    ret[12][0] = weights[scalar] * x_vals[1] * (1-temp)
    ret[13][0] = weights[scalar] * x_vals[2] * (1-temp)
    ret[14][0] = weights[scalar] * (1-temp)
    ret[15][0] = 1
    '''
    ret = numpy.zeros((16,1))
    for i in range(len(weights[0]) - 1):
        if i%num_eq_weights == 0:
            #case where we just multiply by everything above it

            to_tanh = 0
            for k in range(len(x_vals)):
                to_tanh += weights[0][i+k+1]*x_vals[k]
            to_tanh += weights[0][i+len(x_vals)]
            to_append = math.tanh(to_tanh)
            ret[i][0] = to_append

        elif i%num_eq_weights == (num_eq_weights-1):

            to_tanh = 0
            for k in range(len(x_vals)):
                to_tanh += weights[0][i+k+1]*x_vals[k]
            to_tanh += weights[0][i+len(x_vals)]
            to_append = math.tanh(to_tanh)*weights[0][int(i/num_eq_weights)]
            ret[i][0] = to_append

        else:
            to_tanh = 0
            for k in range(len(x_vals)):
                print(i,num_eq_weights)
                print(int(i/num_eq_weights))
                to_tanh += weights[0][i+k+1]*x_vals[k]
            to_tanh += weights[0][i+len(x_vals)]
            to_append = math.tanh(to_tanh) * weights[0][int(i/num_eq_weights)] * x_vals[i%num_eq_weights-1]
            ret[i][0] = to_append
    ret[i][len(weights[0]) - 1] = 1
    '''
    return list(ret)
def generate_x(N,r):
    #add in the x^i values 1
    ret = np.random.uniform(-r, r, (3,N))
    return ret
def generate_y(x):

    y = np.multiply(x[0],x[1]) + x[2]

    return y
def generate_y_3c(x):

    #y = np.sin(np.multiply(x[0],x[1])) + np.cos(x[2])
    y = x[0] - x[1] - x[2]
    return y
def generate_y_3d(x,noise_l):
    y = np.multiply(x[0], x[1]) + x[2] + np.random.uniform(-noise_l,noise_l)
    return y
def d_r_wt(w,N,weight_num,x,lam):
    ret = np.zeros((N, weight_num))
    for i in range(N):
        temp_x = x[:, i]
        temp_x = list(temp_x)
        ret[i, :] = np.array(deriv(w, temp_x)).reshape(weight_num)
    return ret
def d_H_wt(w,N,weight_num,x,lam):
    ret = d_r_wt(w,N,weight_num,x,lam)
    to_concat = np.identity(weight_num)*math.sqrt(lam)
    return np.concatenate((ret,to_concat))

def func_w(w,x_val):
    w = w[0]
    scalar = 0
    ret = 0
    ret += w[scalar]*math.tanh(w[scalar+1] * x_val[0] + w[scalar + 2] * x_val[1] + w[scalar + 3] * x_val[2] + w[scalar + 4])
    scalar = 5
    ret += w[scalar] * math.tanh(w[scalar + 1] * x_val[0] + w[scalar + 2] * x_val[1] + w[scalar + 3] * x_val[2] + w[scalar + 4])
    scalar = 10
    ret += w[scalar] * math.tanh(w[scalar + 1] * x_val[0] + w[scalar + 2] * x_val[1] + w[scalar + 3] * x_val[2] + w[scalar + 4])
    ret += w[15]
    return ret
def r_w(x,y,w,N):
    ret = np.zeros((len(y),1))
    for i in range(N):
        temp_x = x[:, i]
        temp_x = list(temp_x)

        calc_x = func_w(w,temp_x)
        #add them from y[i] - calc[x]

        ret[i][0] = calc_x - y[i]

    return ret

def b_hat(w,N,weight_num,x,lam,y):
    r_w_t = r_w(x,y,w,N)
    to_sub = np.matmul(d_r_wt(w,N,weight_num,x,lam),np.transpose(w))
    ret = np.subtract(r_w_t,to_sub)
    #print("r_w")
    #print(r_w_t)
    ret = np.concatenate((ret,np.zeros((weight_num,1))))
    return ret
def Ls(a,b):
    temp = np.matmul(np.transpose(a),a)
    a_inv = np.linalg.pinv(temp)
    temp = np.matmul(a_inv,np.transpose(a))
    return np.matmul(temp,b)
def l_w_sq(w_prev,N,weight_num,x,lam,y,gamma):

    #take out neg 1 in front of gamme*w transpose
    B = np.concatenate((b_hat(w_prev, N, weight_num, x, lam, y), -1*math.sqrt(gamma) * np.transpose(w_prev)))
    A = np.concatenate((d_H_wt(w_prev, N, weight_num, x, lam), math.sqrt(gamma) * np.identity(weight_num)))
    #print(B.shape,B)
    #print(A.shape,A)
    calc_w = Ls(A,B)
    return calc_w
    #now solve via least squares
def l_w(x,y,w,N,lam):

    temp = np.concatenate((r_w(x,y,w,N),math.sqrt(lam)*np.transpose(w)))
    return np.matmul(np.transpose(temp),temp)[0][0]
def rms(x,y,w,N):
    temp = r_w(x,y,w,N)
    s = np.multiply(temp,temp)
    s = np.sum(s)
    s /= N
    s = math.sqrt(s)
    return s
# need a fcn that will get all
def test_eq(x, y, w,N,lam,gam):
    term = False

    print(l_w(x, y, w, N, lam))

    prev_lw = l_w(x, y, w, N, lam)
    # for loop that will continue until term condition reached
    cnt = 0
    same_cnt = 0
    losses = []
    while not term:
        #print(w)
        cnt += 1
        new_w = np.transpose(l_w_sq(w, N, np.max(np.size(w)), x, lam, y, gam))
        new_lw = l_w(x, y, new_w, N, lam)
        losses.append(new_lw)
        if (abs(new_lw - prev_lw) < .0005):
            same_cnt += 1
        else:
            same_cnt = 0
        print(cnt, prev_lw, new_lw, gam)
        if new_lw < prev_lw:
            gam *= .8
            w = new_w
            prev_lw = new_lw
        else:
            gam *= 2
        if cnt >= 2000 or same_cnt > 20:
            term = True
        if new_lw < .001:
            term = True
        if gam > math.pow(10,20):
            term = True
    return prev_lw,losses,w
def train500_3a():
    N = 500
    num_weights = 16
    val = 1
    w = init_weights(num_weights, val,True)
    eq_num = 3
    x = generate_x(N,1)

    x = np.load("trainx_500.npy")
    np.save("trainx_500", x)
    # x = np.transpose(np.load("train500.npy"))
    y = generate_y(x)
    lam = math.pow(10, -5)
    gam = 1
    # want to see effects of different LM initializations - w/ respect to lamda, gamma, weights
    lamdas = [0,math.pow(10, -7), math.pow(10, -5), 1, 2]
    gammas = [0, math.pow(10, -7), math.pow(10, -5), math.pow(10, -1), 1, 5, 10]
    weight_vals = [.01,.1,1,5]  # -1 means use the rand.uniform fcn
    # to save as a csv -> numpy.savetxt("foo.csv", np.load("conf-L=1000-rle.npy")[1], delimiter=",")
    ret1 = [[(0.0) for i in range(len(weight_vals))] for j in range(len(lamdas))]
    ret1 = np.array(ret1)
    '''
    for i in range(len(lamdas)):
        for j in range(len(weight_vals)):
            to_ret, nvm, new_w = test_eq(x, y, init_weights(num_weights, weight_vals[j],True), N, lamdas[i], gam)
            #temp = rms(x, y, new_w, N)
            ret1[i][j] = to_ret
            #ret2[i][j] = temp
    
    numpy.savetxt("3a1.csv", ret1, delimiter=",")
    #numpy.savetxt("3a2.csv", ret2, delimiter=",")
    print(ret1)
    '''
    res = test_eq(x, y, w, N, lam, gam)
    w = res[2]
    print("final l(w)",res[0])
    print("rms", rms(x, y, w, N))
    matplotlib.pyplot.plot(res[1])
    plt.xlabel("Iteration t")
    plt.ylabel("l(w)")
    plt.title("l(w) calculations over time with lamda = " + str(lam) + " initial gamma = " + str(
        gam) + " inital weights equal to w in range [" + str(val) + "]" )

    plt.show()
    return res[0]
def test100():
    N = 500
    num_weights = 16
    val = -1000
    w = init_weights(num_weights, val,True)
    eq_num = 3
    # x = generate_x(N,1)
    x = np.load("trainx_500.npy")
    np.save("trainx_500", x)
    # x = np.transpose(np.load("train500.npy"))
    y = generate_y(x)
    lam = math.pow(10, -5)
    gam = 1
    # want to see effects of different LM initializations - w/ respect to changing x max and lambda -> want trainig loss/rms test
    lamdas = [0, math.pow(10, -7), math.pow(10, -5), math.pow(10, -1), 1]
    x_vals = [.25,.5,1,5,10]  # -1 means use the rand.uniform fcn
    ret1 = [[0.0 for i in range(len(x_vals))] for j in range(len(lamdas))]
    ret1 = np.array(ret1)
    for i in range(len(lamdas)):
        for j in range(len(x_vals)):
            test_x = generate_x(100,x_vals[j])
            test_y = generate_y(x)
            to_ret, nvm, new_w = test_eq(x, y, init_weights(num_weights, 1,True), N, lamdas[i], gam)
            temp = rms(test_x, test_y, new_w, 100)
            ret1[i][j] = temp

    numpy.savetxt("3b1-rms.csv", ret1, delimiter=",")
    #numpy.savetxt("3b2-tl.csv", ret2, delimiter=",")

#for the part c we just want to do it with a sin fcn or smthing and then call part 3a and 3b with a different generate y fcn
def train500_3c():
    N = 500
    num_weights = 16
    val = 1
    w = init_weights(num_weights, val,True)
    eq_num = 3
    x = generate_x(N,1)

    x = np.load("trainx_500.npy")
    np.save("trainx_500", x)
    # x = np.transpose(np.load("train500.npy"))
    y = generate_y_3c(x)
    lam = math.pow(10, -5)
    gam = 1
    # want to see effects of different LM initializations - w/ respect to lamda, gamma, weights
    lamdas = [0,math.pow(10, -7), math.pow(10, -5), 1, 2]
    gammas = [0, math.pow(10, -7), math.pow(10, -5), math.pow(10, -1), 1, 5, 10]
    weight_vals = [.01,.1,1,5]  # -1 means use the rand.uniform fcn
    # to save as a csv -> numpy.savetxt("foo.csv", np.load("conf-L=1000-rle.npy")[1], delimiter=",")
    ret1 = [[(0.0) for i in range(len(weight_vals))] for j in range(len(lamdas))]
    ret1 = np.array(ret1)
    '''
    for i in range(len(lamdas)):
        for j in range(len(weight_vals)):
            to_ret, nvm, new_w = test_eq(x, y, init_weights(num_weights, weight_vals[j],True), N, lamdas[i], gam)
            #temp = rms(x, y, new_w, N)
            ret1[i][j] = to_ret
            #ret2[i][j] = temp
    numpy.savetxt("3c1-lin.csv", ret1, delimiter=",")
    #numpy.savetxt("3a2.csv", ret2, delimiter=",")
    print(ret1)
    '''
    res = test_eq(x, y, w, N, lam, gam)
    w = res[2]
    print("final l(w)",res[0])
    print("rms", rms(x, y, w, N))

    matplotlib.pyplot.plot(res[1])
    plt.xlabel("Iteration t")
    plt.ylabel("l(w)")
    plt.title("l(w) calculations over time with lamda = " + str(lam) + " initial gamma = " + str(
        gam) + " inital weights equal to w in range [" + str(val) + "]" )

    plt.show()
    return res[0]
def test100_3c():
    N = 500
    num_weights = 16
    val = -1000
    w = init_weights(num_weights, val,True)
    eq_num = 3
    # x = generate_x(N,1)
    x = np.load("trainx_500.npy")
    np.save("trainx_500", x)
    # x = np.transpose(np.load("train500.npy"))
    y = generate_y_3c(x)
    lam = math.pow(10, -5)
    gam = 1
    # want to see effects of different LM initializations - w/ respect to changing x max and lambda -> want trainig loss/rms test
    lamdas = [0, math.pow(10, -7), math.pow(10, -5), math.pow(10, -1), 1]
    x_vals = [.25,.5,1,5,10]  # -1 means use the rand.uniform fcn
    ret1 = [[0.0 for i in range(len(x_vals))] for j in range(len(lamdas))]
    ret1 = np.array(ret1)
    for i in range(len(lamdas)):
        for j in range(len(x_vals)):
            test_x = generate_x(100,x_vals[j])
            test_y = generate_y_3c(x)
            to_ret, nvm, new_w = test_eq(x, y, init_weights(num_weights, 1,True), N, lamdas[i], gam)
            temp = rms(test_x, test_y, new_w, 100)
            ret1[i][j] = temp

    numpy.savetxt("3c2-rms-lin.csv", ret1, delimiter=",")
    #numpy.savetxt("3b2-tl.csv", ret2, delimiter=",")
#partd we want to do if with a different generate y that will add in noise at each point of just add to an array of uniformly distrib noise but
def train500_3d():
    N = 500
    num_weights = 16
    val = 1
    w = init_weights(num_weights, val,True)
    eq_num = 3
    x = generate_x(N,1)

    x = np.load("trainx_500.npy")
    np.save("trainx_500", x)
    # x = np.transpose(np.load("train500.npy"))
    noise_levels = [0,.1,.5,1]
    for nl in noise_levels:

        y = generate_y_3d(x,nl)
        lam = math.pow(10, -5)
        gam = 1
        # want to see effects of different LM initializations - w/ respect to lamda, gamma, weights

        lamdas = [0,math.pow(10, -7), math.pow(10, -5), 1, 2]
        gammas = [0, math.pow(10, -7), math.pow(10, -5), math.pow(10, -1), 1, 5, 10]
        weight_vals = [.01,.1,1,5]  # -1 means use the rand.uniform fcn
        # to save as a csv -> numpy.savetxt("foo.csv", np.load("conf-L=1000-rle.npy")[1], delimiter=",")
        ret1 = [[(0.0) for i in range(len(weight_vals))] for j in range(len(lamdas))]
        ret1 = np.array(ret1)

        for i in range(len(lamdas)):
            for j in range(len(weight_vals)):
                to_ret, nvm, new_w = test_eq(x, y, init_weights(num_weights, weight_vals[j],True), N, lamdas[i], gam)
                #temp = rms(x, y, new_w, N)
                ret1[i][j] = to_ret
                #ret2[i][j] = temp
        '''
        res = test_eq(x, y, w, N, lam, gam)
        w = res[2]
        print("final l(w)", res[0])
        print("rms", rms(x, y, w, N))
        matplotlib.pyplot.plot(res[1])
        plt.xlabel("Iteration t")
        plt.ylabel("l(w)")
        plt.title("l(w) calculations over time with lamda = " + str(lam) + " initial gamma = " + str(
            gam) + " inital weights equal to w in range [" + str(val) + "]" + "for noise level " + str(nl))
        plt.show()
        '''
        numpy.savetxt("testing-3d1-nl-" + str(nl)+ ".csv", ret1, delimiter=",")
    #numpy.savetxt("3a2.csv", ret2, delimiter=",")




    #plt.show()
    return 0
def test100_3d():
    N = 500
    num_weights = 16
    val = -1000
    w = init_weights(num_weights, val,True)
    eq_num = 3
    # x = generate_x(N,1)
    x = np.load("trainx_500.npy")
    np.save("trainx_500", x)
    # x = np.transpose(np.load("train500.npy"))
    noise_levels = [0,.1,.5,1]
    for nl in noise_levels:
        y = generate_y_3d(x,nl)
        lam = math.pow(10, -5)
        gam = 1
        # want to see effects of different LM initializations - w/ respect to changing x max and lambda -> want trainig loss/rms test
        lamdas = [0, math.pow(10, -7), math.pow(10, -5), math.pow(10, -1), 1]
        x_vals = [.25,.5,1,5,10]  # -1 means use the rand.uniform fcn
        ret1 = [[0.0 for i in range(len(x_vals))] for j in range(len(lamdas))]
        ret1 = np.array(ret1)
        for i in range(len(lamdas)):
            for j in range(len(x_vals)):
                test_x = generate_x(100,x_vals[j])
                test_y = generate_y_3c(x)
                to_ret, nvm, new_w = test_eq(x, y, init_weights(num_weights, 1,True), N, lamdas[i], gam)
                temp = rms(test_x, test_y, new_w, 100)
                ret1[i][j] = temp

        numpy.savetxt("testing-3c2-rms-nl-" + str(nl)+ ".csv", ret1, delimiter=",")
        #numpy.savetxt("3b2-tl.csv", ret2, delimiter=",")
#look here at the final training loss, rms, total number of iterations

def debug_test():
    w = init_weights(16,1,False)
    lamda = 0.00001
    gamma = 1
    x = np.ones((3, 1))
    y = generate_y(x)
    temp = l_w(x, y, w, 1, lamda)
    print("temp is",temp)
    w = np.transpose(l_w_sq(w,1,16,x,lamda,y,gamma))
    temp = l_w(x, y, w, 1, lamda)
    print("temp is ", temp)


if __name__ == '__main__':
    train500_3d()
    test100_3d()
    #note w is stored here as a 1,16 matrix so must store the transpose after return




