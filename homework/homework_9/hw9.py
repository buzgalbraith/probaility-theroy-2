# %%
import numpy as np 
from nearest_neighbor import *
from plot_tools import *

# %%
cov=np.zeros([3,3])
cov[0,:]=[100,25,0]
cov[1,:]=[25,400,0]
cov[2,:]=[0,0,.16]
y=np.array([1/np.sqrt(cov[i,i]) for i in range(cov.shape[0])])


# %%
cov_y=cov.copy()
for i in range(cov.shape[0]):
    for j in range(cov.shape[1]):
        cov_y[i,j]=cov_y[i,j]*y[i]*y[j]
cov_y

# %%
y_prime=np.ones((3,1))
# y_prime[:,0]=[1,2,3]
normalize= lambda  y_prime: y_prime/np.linalg.norm(y_prime)
y=normalize(y_prime)
y.T@cov_y@y

# %%
k=2
eig_vals, eig_vectors=np.linalg.eig(cov_y)
print(eig_vals)
print(eig_vectors)
index=np.argsort(eig_vals)[eig_vals.shape[0] -k :]
print(index)
principal_directions=eig_vectors[:, index]
print(principal_directions)

# %%
def compute_nearest_neighbors(train_matrix, testImage):
    calc_distance_metric= lambda x: np.linalg.norm(testImage -x)
    distances=map(calc_distance_metric, train_matrix)
    return np.argmin(list(distances))

compute_nearest_neighbors(train_set, testImage)

# %%
train_set, y_train,test_set,y_test,data=get_data()
testImage = train_set[2, :]


# %%
data=np.concatenate((train_set, test_set), axis=0)
cov=np.cov(data,rowvar=False)
eig_vals, eig_vectors=np.linalg.eig(cov)
flipped_vals=np.flip(np.argsort(eig_vals))
directional_variance=lambda a,cov: a.T@cov@a
directional_variances=[]
for i in range(100):
    directional_variances.append(directional_variance(eig_vectors[:,flipped_vals[i]],cov) )
plt.plot(directional_variances)
#plt.yscale("log")
plt.ylabel("log of variance")
plt.xlabel("principal component number")
plt.title("log of directional variance plotted against principal component number")

# %%
data=np.concatenate((train_set, test_set), axis=0)
cov=np.cov(data,rowvar=False)
eig_vals, eig_vectors=np.linalg.eig(cov)
flipped_vals=np.flip(np.argsort(eig_vals))
directional_variance=lambda a,cov: a.T@cov@a
directional_variances=[]
for i in range(100):
    directional_variances.append(directional_variance(eig_vectors[:,flipped_vals[i]],cov) )
plt.plot(directional_variances)
#plt.yscale("log")
plt.ylabel("log of variance")
plt.xlabel("principal component number")
plt.title("log of directional variance plotted against principal component number")
principal_directions=np.ones((4096,10))
for i in range(10):
    principal_directions[:,i]=eig_vectors[:,flipped_vals[i]]
plot_image_grid(principal_directions.T, "principal_directions")

# %%
plot_image_grid(principal_directions.T, "principal_directions")

# %%
train_set.shape

# %%
def compute_nearest_projected_neighbors(train_matrix, testImage,k):
    top_4_directions=get_top_k_principle_directions(k)
    projected_test_data=testImage@top_4_directions
    projected_training_data=train_matrix@top_4_directions
    calc_distance_metric= lambda x: np.linalg.norm(projected_test_data -x)
    distances=map(calc_distance_metric, projected_training_data)
    return np.argmin(list(distances))

compute_nearest_projected_neighbors(train_set, testImage,5)

# %%
k=4
def get_top_k_principle_directions(k):
    principal_directions=np.ones((4096,k))
    for i in range(k):
        principal_directions[:,i]=eig_vectors[:,flipped_vals[i]]
    return principal_directions
def plot_nn_using_dimension_k(k):
    imgs = []
    estLabels = []
    for i in range(test_set.shape[0]):
        testImage = test_set[i, :]
        nnIdx = compute_nearest_projected_neighbors(train_set, testImage,k)
        imgs.extend( [testImage, train_set[nnIdx,:]] )
        estLabels.append(y_train[nnIdx])


    row_titles = ['Test','Nearest']
    col_titles = ['%d vs. %d'%(i,j) for i,j in zip(y_test, estLabels)]
    plot_tools.plot_image_grid(imgs,
                    "Image-NearestNeighbor",
                    (64,64), len(test_set),2,True,row_titles=row_titles,col_titles=col_titles)
plot_nn_using_dimension_k(20)

# %%



