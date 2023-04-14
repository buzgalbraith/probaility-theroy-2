import numpy as np
from sklearn.datasets import fetch_olivetti_faces
import plot_tools


def compute_nearest_neighbors(train_matrix, testImage):
    calc_distance_metric= lambda x: np.linalg.norm(testImage -x)
    distances=map(calc_distance_metric, train_matrix)
    return np.argmin(list(distances))

def get_data():
    test_idx = [1,  87,  94, 78]

    data = fetch_olivetti_faces()
    targets = data.target
    data = data.images.reshape((len(data.images), -1))

    train_idx = np.array(list(set(list(range(data.shape[0]))) - set(test_idx) ) )

    train_set = data[train_idx ]
    y_train = targets[train_idx] 
    test_set = data[np.array(test_idx)]
    y_test = targets[ np.array(test_idx)]
    return train_set, y_train,test_set,y_test,data



def main() :
    train_set, y_train,test_set,y_test,data=get_data()
    imgs = []
    estLabels = []
    for i in range(test_set.shape[0]):
        testImage = test_set[i, :]
        nnIdx = compute_nearest_neighbors(train_set, testImage)
        imgs.extend( [testImage, train_set[nnIdx,:]] )
        estLabels.append(y_train[nnIdx])


    row_titles = ['Test','Nearest']
    col_titles = ['%d vs. %d'%(i,j) for i,j in zip(y_test, estLabels)]
    plot_tools.plot_image_grid(imgs,
                    "Image-NearestNeighbor",
                    (64,64), len(test_set),2,True,row_titles=row_titles,col_titles=col_titles)
    
    
    
if __name__ == "__main__" :
    main()