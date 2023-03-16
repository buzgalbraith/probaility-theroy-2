import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sys
path=sys.argv[1]
data=pd.read_csv(path ,sep=";")
data=data["ap_hi"]
temp=np.random.choice(data, int(len(data)*.01))
bias= lambda x: np.mean(x)-np.mean(data)
def find_bias_variance(start, step):
    out=[[],[]]
    for i in np.arange(start, 1, step):
        x=np.random.choice(data, int(len(data)*i))
        out[0].append(bias(x))
        out[1].append(np.var(x))
    plt.plot(np.arange(start, 1, step), out[0])
    plt.xlabel("percentage  sampled")
    plt.ylabel("bias")
    plt.title("bias of sample for different proportions of the sample")
    plt.show()
    plt.plot(np.arange(start, 1, step), out[1])
    plt.xlabel("percentage  sampled")
    plt.ylabel("variance")
    plt.title("variance of  sample for different proportions of the sample")
    plt.show()
    return out
def monte_carlo(trials, sample_sizes,c=[.25,.5,1,10], show_dist=False, show_bound=True):
    for sample_size in sample_sizes:
        bias_out=[]
        out=[]
        for trial in range(trials):
            x=np.random.choice(data, sample_size)
            bias_out.append(bias(x))
        if(show_dist):
            weights = np.ones_like(bias_out) / trials
            plt.hist(bias_out, weights=weights, label="sample size={0}".format(sample_size))    
            plt.xlabel("bias")
            plt.ylabel("probability")
            print(np.median(bias_out))
            plt.title("distobution of bias for difent sample size")
        elif(show_bound):
            temp_bias=[]
            for bound in c:
                bias_out=np.array(bias_out)
                temp_bias.append(sum(abs(bias_out)>=bound)/trials)
            plt.plot(c,temp_bias, label="sample size={0}".format(sample_size))
            plt.xlabel("value of c")
            plt.ylabel("probability |x-mu|>c")
            plt.title("probability sample bias bellow different thresholds for different sample size")
    plt.legend()
    plt.show()
def show_chevchevs_bound():
    a=np.linspace(1,1000,10000)
    plt.plot(a,np.var(data)/(a**2))
    plt.yscale('log')
    plt.ylabel("log of bound")
    plt.xlabel("c")
    plt.title("bound given by Chebyshev's inequality on given data")
    plt.show()
b=np.var(data)
a=np.sqrt(b)
print(a)
print(b)
print(b/(a**2))

#find_bias_variance(.01,.01)
#monte_carlo(1000, [500,1000,5000, 10000, 20000, 30000,40000,50000, 60000,70000], show_dist=False, show_bound=1)
#show_chevchevs_bound()