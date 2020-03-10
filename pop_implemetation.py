import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.metrics import pairwise_distances
# X: n rows (examples) x p attribures
# y: true class
# cont_attr: indexes of continuous attributes. If None,
# then all the p attributes are assumed to be continued

# NB Binary classification
def pop_instance_training(X, y,  metric='euclidean'):

    isborder = np.zeros(X.shape)
    mindistance = np.zeros(X.shape[1])
    maxdistance = np.zeros(X.shape[1]) # max distance between classes in one attribute

    attribute2innerborder = dict(zip(list(range(X.shape[1])), [[] for x in range(X.shape[1])]))
    attribute2outerborder = dict(zip(list(range(X.shape[1])), [[] for x in range(X.shape[1])]))

    attribute2innerborder_value = dict(zip(list(range(X.shape[1])), [[] for x in range(X.shape[1])]))
    attribute2outerborder_value = dict(zip(list(range(X.shape[1])), [[] for x in range(X.shape[1])]))


    # If weakness(x) = #attributes --> x is not a border instance
    weakness = np.zeros(X.shape[0])

    # Divide examples according to class label
    iC1 = np.where(y==np.unique(y)[0])[0]
    iC2 = np.where(y==np.unique(y)[1])[0]

    # For each attribute
    for i in range(X.shape[1]):
        #col_dis = pairwise_distances(np.reshape(np.array(X[:,i]), (X.shape[0], 1)),
        #                             metric=metric)

        Xc1 = np.reshape(X[iC1, i], (X[iC1].shape[0], 1))
        Xc2 = np.reshape(X[iC2, i], (X[iC2].shape[0], 1))

        pointwise_dis = cdist(Xc1, Xc2)
        min_val = np.argwhere(pointwise_dis==np.min(pointwise_dis))
        max_val = np.argwhere(pointwise_dis==np.max(pointwise_dis))

        isborder[iC1[min_val[0][0]], i] = -1 # -1 --> inner border
        isborder[iC2[min_val[0][1]], i] = -1
        isborder[iC1[max_val[0][0]], i] = 1 # 1 --> outer border
        isborder[iC2[max_val[0][1]], i] = 1

        attribute2innerborder[i]  = attribute2innerborder[i]+[iC1[min_val[0][0]]]
        attribute2innerborder[i] = attribute2innerborder[i]+[iC2[min_val[0][1]]]

        attribute2outerborder[i] = attribute2outerborder[i]+ [iC1[max_val[0][0]]]
        attribute2outerborder[i] = attribute2outerborder[i] + [iC2[max_val[0][1]]]

        attribute2innerborder_value[i] = attribute2innerborder_value[i] + [X[iC1[min_val[0][0]], i]]
        attribute2innerborder_value[i] = attribute2innerborder_value[i] + [X[iC2[min_val[0][1]], i]]

        attribute2outerborder_value[i] = attribute2outerborder_value[i] + [X[iC1[max_val[0][0]], i]]
        attribute2outerborder_value[i] = attribute2outerborder_value[i] + [X[iC2[max_val[0][1]], i]]


        mindistance[i] = np.min(pointwise_dis)
        maxdistance[i] = np.max(pointwise_dis)

    #     max_dis = np.max(col_dis, axis=1)
    #     np.fill_diagonal(col_dis, np.inf)
    #     min_dis = np.min(col_dis, axis=1)
    #
    #     # For each example, see if it is border
    #     for j in range(X.shape[0]):
    #         cj = y[j]
    #         c_opp_i = np.where(y != cj)[0]
    #         dis_opp = min_dis[c_opp_i]
    #
    #         if np.min(dis_opp) == min_dis[j]: # Is border
    #             isborder[j,i]=1
    #             mindistance[i] = np.min((mindistance[i], min_dis[j]))
    #         else:
    #             weakness[j] = weakness[j]+1
    #             maxdistance[i] = np.max(maxdistance[i], max_dis[j])
    #
    return(mindistance,
            maxdistance,
            isborder,
            attribute2outerborder,
            attribute2innerborder,
           attribute2outerborder_value,
           attribute2innerborder_value)


# z: single attribute of one example
# border_val: attribute values for the 2 border example
def check_is_border(z, border_val, inner=True):

    is_b = 'NoBorder'
    # It's border if for at least one attribute, its distance from
    # one of the two inner border is less then the distance between the border instances
    if len(border_val)> 2:
        print('More than 2 border instance')
        exit(1)
    if inner:
        if z >= np.min(border_val) and z <= np.max(border_val):
            is_b = 'InnerBorder'
    else:
        if z <= np.min(border_val) or z >= np.max(border_val):
            is_b = 'OuterBorder'

    return is_b
