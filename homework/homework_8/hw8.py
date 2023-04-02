# %%
import numpy as np
import scipy.stats as stats
import scipy as sp 
import matplotlib.pyplot as plt
import scipy.stats as stats

# %% [markdown]
# ## question 1 a

# %%
p=np.zeros([3,3])
p[0,0]=1
p[1,1]=1/np.sqrt(2)
p[1,2]=1/np.sqrt(2)
p[2,1]=1/np.sqrt(2)
p[2,2]=-1/np.sqrt(2)
eigen_matirx=np.zeros([3,3])
eigen_matirx[0,0]=1
eigen_matirx[1,1]=1/2
print(p,"\n", eigen_matirx)
covariance_matrix=p@eigen_matirx@np.linalg.inv(p)
print(covariance_matrix)

## so only psd

# %% [markdown]
# ## question 1 b and c. 

# %%
a=[0,(1/np.sqrt(2)),(1/np.sqrt(2))]
def objective(a):
    a_1, a_2, a_3 = a
    return a_1**2 + 0.25*(a_2**2 + a_3**2) + 0.5*a_2*a_3 
objective(a) ##.5 
a=[1,0,0]
objective(a) ##!1
a=[0,1/np.sqrt(2),1/np.sqrt(2)] ##.66 
print(objective(a) )
a=[0,1/np.sqrt(2),-1/np.sqrt(2)] ##.66 
objective(a) 

# %% [markdown]
# ## question 2. 

# %%
covariance_matrix=np.zeros([3,3])
covariance_matrix[0,:]=[100,-80,10]
covariance_matrix[1,:]=[-80,81,50]
covariance_matrix[2,:]=[10,50,100]
covariance_matrix

# %%
def find_variance(a, covariance_matrix):
    return np.dot(a,covariance_matrix)@a
a=np.array([1,1,0])
print(find_variance(a,covariance_matrix))
a=np.array([1,0,1])
print(find_variance(a,covariance_matrix))
a=np.array([0,1,1])
print(find_variance(a,covariance_matrix))

# %% [markdown]
# ## question 3a. 

# %%
sigma_signal=50
sigma_noise=np.identity(2)*1
x=stats.norm(loc=0, scale=sigma_signal)
z=stats.multivariate_normal(mean=[0,0],cov=sigma_noise)
v=np.array([1/np.sqrt(2),1/np.sqrt(2)])
def y(n):
    ellement_wise_multiply = lambda v,n: np.array([v*x_i for x_i in x.rvs(n)])
    return ellement_wise_multiply(v,n)+z.rvs(size=n)
samples=y(10000)
plt.scatter(samples[:,0],samples[:,1])
plt.show()
plt.hist(samples)


# %% [markdown]
# ## question 3b. 

# %%
sigma_signal=1
sigma_noise=np.identity(2)*50
x=stats.norm(loc=0, scale=sigma_signal)
z=stats.multivariate_normal(mean=[0,0],cov=sigma_noise)
v=np.array([1/np.sqrt(2),1/np.sqrt(2)])
def y(n):
    ellement_wise_multiply = lambda v,n: np.array([v*x_i for x_i in x.rvs(n)])
    return ellement_wise_multiply(v,n)+z.rvs(size=n)
samples=y(10000)
plt.scatter(samples[:,0],samples[:,1])
plt.xlabel("$y_1$ value")
plt.ylabel("$y_2$ value")
plt.show()
plt.hist(samples)


# %% [markdown]
# ## question 4. 

# %%
from data import findata_tools
df = findata_tools.load_dataframe(r'/home/buzgalbraith/work/school/spring_2023/probaility-theroy-2-2023/homework_code/homework_8/data/stockprices.csv')

# %%
def center_data(df):
    center_data=df.copy()
    for col in df.columns:
        center_data[col]=df[col]-np.mean(df[col])
    return center_data
centerd_data=center_data(df)


# %%
def sample_mean(x):
    out=0
    for x_i in x:
        out=out+x_i
    return out/len(x)
def sample_variance(x):
    inner_term=x-sample_mean(x)
    return sample_mean(inner_term**2)
def sample_covariance(x,y):
    out=0
    expected_x=sample_mean(x)
    expected_y=sample_mean(y)
    for i in range(len(x)):
        out+=(x[i]-expected_x)*(y[i]-expected_y)
    return out/(len(x)-1)
def sample_covariance_matrix(D):
    covariance_matrix=[]
    for x in D:
        temp=[]
        for y in D:
            temp.append(sample_covariance(D[x],D[y]))
        covariance_matrix.append(temp)
    return np.array(covariance_matrix)
covariance_matrix=sample_covariance_matrix(centerd_data)

# %%
eigen_values, eigen_vectors=np.linalg.eig(covariance_matrix)
eigen_values.shape
top_2=np.argsort(eigen_values)[-2:]
top_2_eigenvalues=eigen_values[top_2]
top_2_eigenvectors=eigen_vectors[:,top_2]
print(top_2_eigenvectors.shape)

# %%
indecies=np.argsort(top_2_eigenvectors[:,0])[-2:]
top_2_coeficents=centerd_data.values[0,:][indecies]
top_2_companies=centerd_data.columns[np.argsort(top_2_eigenvectors[:,0])]
findata_tools.pretty_print_latex(top_2_eigenvectors[:,0],df.columns)

# %%
indecies=np.argsort(top_2_eigenvectors[:,1])[-2:]
top_2_coeficents=centerd_data.values[0,:][indecies]
top_2_companies=centerd_data.columns[np.argsort(top_2_eigenvectors[:,0])]
findata_tools.pretty_print_latex(top_2_eigenvectors[:,1],df.columns)

# %%
i=range(centerd_data.shape[0])
for col in centerd_data:
    if(col =="GOOG"):
        plt.plot(i, centerd_data[col], color="r",label=col)
    elif(col=="AMZN"):
        plt.plot(i, centerd_data[col], color="b",label=col)
    else:
        plt.plot(i, centerd_data[col], color="g")
plt.legend()
plt.xlabel("days since start of observation period")
plt.ylabel("centerd closing price")
plt.show()

# %%
def standardize_data(true_data):
    data=true_data.copy()
    for col in data:
        data[col]=( data[col] - sample_mean(data[col]) ) / np.sqrt(sample_variance(data[col]))
    return data
standardized_data=standardize_data(centerd_data)
corelation_matrix=sample_covariance_matrix(standardized_data)

# %%
eigen_values, eigen_vectors=np.linalg.eig(corelation_matrix)
eigen_values.shape
top_2=np.argsort(eigen_values)[-2:]
top_2_eigenvalues=eigen_values[top_2]
top_2_eigenvectors=eigen_vectors[:,top_2]
top_2_eigenvectors.shape

# %%
indecies=np.argsort(top_2_eigenvectors[:,0])[-2:]
top_2_coeficents=centerd_data.values[0,:][indecies]
top_2_companies=centerd_data.columns[np.argsort(top_2_eigenvectors[:,0])]
findata_tools.pretty_print_latex(top_2_eigenvectors[:,0],df.columns)

# %%
indecies=np.argsort(top_2_eigenvectors[:,1])[-2:]
top_2_coeficents=centerd_data.values[0,:][indecies]
top_2_companies=centerd_data.columns[np.argsort(top_2_eigenvectors[:,0])]
findata_tools.pretty_print_latex(top_2_eigenvectors[:,1],df.columns)

# %% [markdown]
# ## question 4c. 

# %%
def stock_risk(data, alpha):
    data=center_data(data)
    covariance_matrix=sample_covariance_matrix(data)
    return np.dot(alpha,covariance_matrix )@alpha
alpha=np.ones((1,18)).flatten()*100
alpha[:4]=alpha[:4]*2
stock_risk(df, alpha)


# %% [markdown]
# ## question 4d. 

# %%
data=center_data(df)
covariance_matrix=sample_covariance_matrix(data)
variance=alpha.T@covariance_matrix@alpha
y=stats.norm(loc=0,scale=variance)
(1-y.cdf(-1000))*100


