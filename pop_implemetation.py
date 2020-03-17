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

        pointwise_dis = cdist(Xc1, Xc2, metric=metric)
        min_val = np.argwhere(pointwise_dis == np.min(pointwise_dis))
        max_val = np.argwhere(pointwise_dis == np.max(pointwise_dis))

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

    return(mindistance,
            maxdistance,
            isborder,
            attribute2outerborder,
            attribute2innerborder,
           attribute2outerborder_value,
           attribute2innerborder_value)

# X (normalized or standardized if euclidean is used)
# metric can be:
# - euclidean
# - cosine
# - minkowski with r=1
# - chebyshev (sup=Chebyshev, https://en.wikipedia.org/wiki/Chebyshev_distance)
def find_multidim_borders(X, y, metric='euclidean', r=None):

    isborder = np.zeros(X.shape[0])

    cl = np.unique(y)
    iCl1 = np.where(y==cl[0])[0]
    iCl2 = np.where(y==cl[1])[0]

    X1 = X[iCl1,:]
    X2 = X[iCl2, :]

    if r is not None:
        d = cdist(X1, X2, metric=metric)
    else:
        d = cdist(X1, X2, metric=metric, p=r)
    min_d = d.min() # min (def. euclidean) distance between two examples of two different classes
    max_d = d.max() # max (def. euclidean) distance between two examples of two different classes

    i_inner_borders = np.argwhere(d == min_d)

    for i in range(len(i_inner_borders)):
        isborder[iCl1[i_inner_borders[i][0]]] = -1
        isborder[iCl2[i_inner_borders[i][1]]] = -1

    i_outer_borders = np.argwhere(d == max_d)
    for i in range(len(i_outer_borders)):
        isborder[iCl1[i_outer_borders[i][0]]] = 1
        isborder[iCl2[i_outer_borders[i][1]]] = 1

    return(isborder, min_d, max_d)

# z: single attribute of one example
# border_val: X borders
# d: distance between the X borders
def check_is_border(z, border_val, inner=True):

    is_b = 'NoBorder'
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


def check_is_multidim_border(X, z, isborder, min_d, max_d, metric='euclidean', r=None):
    is_b = 'NoBorder'

    # It's inner border if its between the two inner border examples, i.e.:
    # the distances between the new instances and the inner borders is less then the distance between the borders itself
    i_inner = np.where(isborder == -1)[0]

    if r is None:
        d_inner = cdist(z.reshape(1,-1), X[i_inner, :], metric=metric)
    else:
        d_inner = cdist(z.reshape(1, -1), X[i_inner, :], metric=metric, p=r)

    if d_inner[0][0] < min_d:
        if d_inner[0][1] < min_d:
            return 'InnerBorder'

    # If it is not inner check if it is outer
    # i.e. its distance with one of the two outer is greater than max_d
    i_outer = np.where(isborder == 1)[0]
    if r is None:
        d_outer = cdist(z.reshape(1,-1), X[i_outer, :], metric=metric)
    else:
        d_outer = cdist(z.reshape(1, -1), X[i_outer, :], metric=metric, p=r)

    if d_outer[0][0] > max_d:
        return 'OuterBorder'
    if d_outer[0][1] > max_d:
        return 'OuterBorder'

    return is_b

# is_rel: 1 if the i example is reliable, 0 otherwise
# is_correct: 1 if the i example has been correctly classified
# return: H(S) = S1/S*H(S1) - S2/S*H(S2)
# S1 unrel examples, S2 rel examples
# H: Entropy

def get_gain(is_rel, is_correct):

    i_rel = np.where(is_rel == 1)[0]
    i_unrel = np.where(is_rel == 0)[0]

    # Prob of correctness in unrel and rel exampls
    pc_unrel = len(np.where(is_correct[i_unrel]==1)[0])/len(i_unrel)
    pc_rel = len(np.where(is_correct[i_rel]==1)[0])/len(i_rel)

    # Prob of correctness in the total examples
    pc = len(np.where(is_correct==1)[0])/len(is_correct)

    h_unrel = -pc_unrel*np.log2(pc_unrel)-(1-pc_unrel)*np.log2(1-pc_unrel)
    h_rel = -pc_rel*np.log2(pc_rel)-(1-pc_rel)*np.log2(1-pc_rel)

    h_tot = -pc*np.log2(pc)-(1-pc)*np.log2(1-pc)

    gain = h_tot - (len(i_unrel)/len(is_rel))*h_unrel - (len(i_rel)/len(is_rel))*h_rel

    return (gain, h_tot, h_unrel, h_rel)