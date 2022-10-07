import numpy as np


def add_flat_labels(soundings, labels):
    """
    Returns three masks for identification of flat soundings above, below, and in atmosphere
    """
    mean = soundings.mean(axis=1)
    
    above_mask = (mean > 0.99) & (labels==2)
    below_mask = (mean < 0.01) & (labels==2)
    
    labels = np.where(above_mask, 0, labels)
    labels = np.where(below_mask, 1, labels)

    np.savetxt("../data/labels.txt", labels)

    return labels

    # Possible fix: Flat cluster being identified as below and above atmosphere

    """mean = soundings.mean(axis=1)

    labels2 = np.copy(labels)         
    
    above_mask = (mean > 0.99) & (labels==2)
    below_mask = (mean < 0.01) & (labels==2)
    
    labels2 = np.where(above_mask, 0, labels2)
    labels2 = np.where(below_mask, 1, labels2)

    np.savetxt("../data/labels.txt", labels2)

    return labels2"""

def merge_labels(soundings, labels, grad, polyres, poly_coeffs, curv):
    """
    Merges cluster labels into final major classes by considering average features over a cluster

    Inputs
    labels            Labels as outputted by BIRCH algorithm
    grad              gradient calculated by create_features function
    polyres           polynomial residual feature array as outputted by create_features function    
    poly_coeffs       polynomial coeffiecients feature array    
    curv              curvature feature array

    Outputs
    labels            Merged major labels corresponding to final classes (eg periodic, high gradient)
    """
    flat = []
    periodic = []
    positive_curvature = []
    negative_curvature = []
    high_gradient = []
    small_gradient = []

    for c in np.unique(labels):
        
        std = soundings[labels==c].std(axis=1).mean()
        avg_pres1 = polyres[labels == c, 1].mean()
        avg_pcoeff = poly_coeffs[labels == c, 0].mean()
        avg_curv = curv[labels == c, 0].mean() + curv[labels == c, 1].mean()
        avg_grad = sum([grad[labels == c, i].mean() for i in range(1,3)])
        
        per_cond = []
        per_cond.append((np.array([curv[labels == c, i].mean() for i in range(3)]).min() * 
                         np.array([curv[labels == c, i].mean() for i in range(3)]).max()) < 0)
        per_cond.append((np.array([grad[labels == c, i].mean() for i in range(4)]).min() * 
                         np.array([grad[labels == c, i].mean() for i in range(4)]).max()) < 0)
        per_cond.append(avg_pres1 > 2 or avg_pcoeff < -5 or avg_pcoeff > 3.5)
        
        pos_curv_cond = []
        pos_curv_cond.append(avg_pres1 > 7 or avg_pcoeff < -5 or avg_pcoeff > 3.5)
        pos_curv_cond.append(avg_curv > 0)
        
        if (std < 0.008) and not all(per_cond):
            flat.append(c)
            
        elif all(per_cond):
            periodic.append(c)
        
        elif (avg_pres1 > 5 or avg_pcoeff < -5 or avg_pcoeff > 3.5) and avg_curv > 0:
            positive_curvature.append(c)
            
        elif avg_curv > 2:
            positive_curvature.append(c)
            
        elif (avg_pres1 > 5 or avg_pcoeff < -5 or avg_pcoeff > 3.5) and avg_curv < 0:
            negative_curvature.append(c)

        elif avg_curv < -4 and avg_grad > 2:
            negative_curvature.append(c)
            
        elif avg_grad > 3:
            high_gradient.append(c)
            
        else:
            small_gradient.append(c)
    
    labels2 = np.where(np.in1d(labels, np.array(flat)), 2, labels)
    labels2 = np.where(np.in1d(labels, np.array(periodic)), 5, labels2)
    labels2 = np.where(np.in1d(labels, np.array(negative_curvature)), 4, labels2)
    labels2 = np.where(np.in1d(labels, np.array(positive_curvature)), 6, labels2)
    labels2 = np.where(np.in1d(labels, np.array(high_gradient)), 3, labels2)
    labels2 = np.where(np.in1d(labels, np.array(small_gradient)), 7, labels2)
    
    return labels2