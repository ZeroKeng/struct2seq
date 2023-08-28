import torch
import numpy as np
import scipy
import scipy.spatial
import math

def protein_featurize(batch):
    alphabet = 'ACDEFGHIKLMNPQRSTVWY-'
    B = len(batch)
    lengths = [len(b['seq']) for b in batch]
    L = max(lengths)
    X = torch.zeros(B,L,4,3)
    S = torch.zeros(B,L)
    for i, b in enumerate(batch):
        l = len(b['seq'])
        # structure featurization
        #join all atom(N,CA,C,O) as array along 1 axis 
        x = np.stack([b['coords'][atom] for atom in ['N','CA','C','O']],axis=1) #for each atom[l,3] -> [l,4,3]
        # for a tensor shaped with [l,4,3], pad into [L,4,3], axis 0 is to pad, 1 and 2 intact 
        x_pad = np.pad(x, pad_width=[[0,L-l],[0,0],[0,0]], mode='constant', constant_values = (np.nan))
        X[i,:,:,:] = torch.Tensor(x_pad)
        
        # sequence featurization
        s = np.array([alphabet.index(aa) for aa in b['seq']])
        s_pad = np.pad(s, pad_width=[[0,L-l]],mode='constant',constant_values = (20))
        S[i,:] = torch.Tensor(s_pad)
        S = S.to(torch.long)
        
    #mask
    isnan = torch.isnan(X)
    mask = torch.isfinite(torch.sum(X, dim = (2,3)),).float()
    X[isnan] = 0.
    #print(X.shape)
    X = torch.stack([torch.Tensor(get_coords6d(np.array(i))) for i in X ])
    #X = torch.cat(torch.Tensor([get_coords6d(np.array(X[i])) for i in range(X.shape[0])]))
    X = X.reshape((X.shape[0],X.shape[1],-1))
    return X, S, mask

#####This is the 6D coordinates that encode all the protein structure information
def get_dihedrals(a, b, c, d):
    # Ignore divide by zero errors
    np.seterr(divide='ignore', invalid='ignore')
    
    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c
    
    b1 /= np.linalg.norm(b1, axis=-1)[:,None]
    v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1

    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1, v)*w, axis=-1)
    return np.arctan2(y, x)

# calculate planar angles defined by 3 sets of points
def get_angles(a, b, c):

    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:,None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:,None]

    x = np.sum(v*w, axis=1)

    return np.arccos(x)

# get 6d coordinates from x,y,z coords of N,Ca,C atoms
def get_coords6d(xyz, k=30, dmax=30.0, normalize=True):

    nres = xyz.shape[0]

    # three anchor atoms
    N  = xyz[:,0]
    Ca = xyz[:,1]
    C  = xyz[:,2]

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
    # fast neighbors search to collect all
    # Cb-Cb pairs within dmax
    # kdCb = scipy.spatial.cKDTree(Cb)
    # indices = kdCb.query_ball_tree(kdCb, dmax)
    
    kdtree = scipy.spatial.cKDTree(Cb)
    distances, indices = kdtree.query(Cb,k=k+1)
    indices = indices[:,1:]
    # indices of contacting residues
    #idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i]]).T
    idx0 = idx[0]
    idx1 = idx[1]
    #print(idx0,idx0.shape)
    # Cb-Cb distance matrix
    #dist6d = np.full((nres, nres), dmax).astype(float)
    #dist6d[idx0,idx1] = np.linalg.norm(Cb[idx1]-Cb[idx0], axis=-1)
    dist6d = distances[:,1:]
    #print(dist6d.shape)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    #omega6d = np.zeros((nres, k))
    #omega6d[idx0,idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])
    omega6d = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])
    #print(omega6d,omega6d.shape)
    omega6d = np.reshape(omega6d, (nres,k))
    
    # matrix of polar coord theta
    #theta6d = np.zeros((nres, k))
    #theta6d[idx0,idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])
    theta6d = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])
    theta6d = np.reshape(theta6d, (nres,k))

    # matrix of polar coord phi
    #phi6d = np.zeros((nres, k))
    #phi6d[idx0,idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])
    phi6d = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])
    phi6d = np.reshape(phi6d, (nres,k))
    
    # Normalize all features to [-1,1]
    if normalize:
        # [4A, 20A]
        dist6d = (dist6d / dmax*2) - 1
        # [-pi, pi]
        omega6d = omega6d / math.pi
        # [-pi, pi]
        theta6d = theta6d / math.pi
        # [0, pi]
        phi6d = (phi6d / math.pi*2) - 1

    coords_6d = np.stack([dist6d,omega6d,theta6d,phi6d],axis=-1)

    return coords_6d

