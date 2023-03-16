import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
path='/mnt/c/Users/buzga/Desktop/School/grad_school/spring_2023/prob_stats_2/homework/homework_2/ANSUR_II_MALE_Public.csv'
data=pd.read_csv(path,encoding='latin-1')
print(data.head())
## ols= rho_{x,y}v(y)s(x)+m(x)
def sample_average(x, numpy=False):
    if(numpy):
        return np.mean(x)
    return sum(x)/len(x)
def sample_variance(x,numpy=False):
    if(numpy):
        return np.var(x)
    return sum((x-sample_average(x))**2)/len(x)
def sample_std(x, numpy=False):
    if(numpy):
        return np.std(x)
    return sample_variance(x)**(1/2)
def standardize(x):
    a=x-sample_average(x)
    b=sample_variance(x)**(1/2)
    return (x-sample_average(x))/(sample_variance(x)**(1/2))
def sample_cov(x,y,numpy=False):
    if(numpy):
        return np.cov(x,y)[0,1]
    a=x-sample_average(x)
    b=y-sample_average(y)
    return sample_average(a*b)
def corelation_coeficent(x,y,numpy=False):
    if(numpy):
        return np.corrcoef(x,y)[0,1]
    return sample_cov(x,y)/(sample_std(y)*sample_std(x))
def ols(x,y):
    alpha=(corelation_coeficent(x,y)*sample_std(y))/sample_std(x)
    beta= sample_average(y)-(sample_average(x) *corelation_coeficent(x,y)*sample_std(y))/sample_std(x) 
    return alpha*x+beta, alpha, beta
def sample_r_squared(x,y):
    return sample_variance(ols(x,y)[0])/sample_variance(y)
#Compute the OLS estimator of weight given height. x=height, y=weight
#Compute the sample covariance between the height and the residual of the OLS estimator. Are they correlated?
    
working_x=data["Heightin"]
working_y=data["Weightlbs"]
print("question 1")
linear,alpha, beta=ols(working_x,working_y)
print("the coefcients are alpha={0}, beta={1}".format(alpha, beta))
print("__________"*25)
print("question 2")
outcome_q2=sample_cov(working_x,working_y-linear )
print("the sample covariance between the height and the residual of the OLS estimator is {0}, whcih is very close\
 to 0 meaning that it is likely height and the residual of the OLS estimator are uncorelated".format(outcome_q2))
print("__________"*25)
print("question 3")
#Compute the sample variance of the weight, the OLS estimator of the weight, and the residual. What relationship do you find among these three values?
v_1=sample_variance(working_y)
v_2=sample_variance(linear)
v_3=sample_variance(working_y-linear)
print("The sample variance of the weight is {0}\nThe sample variance of the OLS estimator of the weight is {1}\nthe sample varinace of the risidual is {2}\nthe sum of The sample variance of the OLS estimator of the weight and the sample varinace of the risidual is {3} which is equal toThe sample variance of the weight".format(v_1,v_2,v_3,v_2+v_3))
print("__________"*25)
print("question 4")
'''Compute and compare the sample coefficient of determination (using its definition
as the fraction of the variance explained by the linear estimator), and compare it to
the squared sample correlation coefficient'''
r_2=sample_r_squared(working_x, working_y)
rho=corelation_coeficent(working_x,working_y)
print("The sample the sample coefficient of determination is {0}\nThe squared sample correlation coefficient is {1}\n thus we can see they are very nearly equal".format(r_2, rho**2))