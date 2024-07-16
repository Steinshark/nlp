import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.linalg import svd, norm


def truncated_svd(A,r=None,tol=10**-6):
    '''
    Computes the truncated SVD of A. If r is None or equals the number of nonzero singular values, it is the compact SVD.
    Parameters:
        A: the matrix
        r: the number of singular values to use 
        tol: the tolerance for zero
    Returns:
        U - the matrix U in the SVD
        s - the diagonals of Sigma in the SVD
        Vh - the matrix V^H in the SVD
    '''
    #Initialize things
    m,n = A.shape

    #Find eigenvalues and eigenvectors of A^H A
    eigs, vr = la.eig(A.conj().T.dot(A)) 
    #Find singular values
    sigs = np.sqrt(eigs) 
    #Find how many singular values are nonzero
    mask = sigs > tol 
    num_eigs = np.sum(mask)
    if (r==None):
        #Return compact SVD
        r=num_eigs
    elif (num_eigs < r):
        print 'less nonzero eigenvalues than given size'
        return
    #Initialize things
    U = np.empty((m,r))
    s = np.zeros(r)
    V = np.empty((n,r))
    
    
    #Sort the singular values and only keep the greatest r
    sorted_index = np.argsort(sigs)[::-1]
    sigs = sigs[sorted_index]
    s[:r] = sigs[:r]
    #Keep eigenvectors matching the order of the singular values
    vr = vr[:,sorted_index]
    
    #Calculate V
    #Only keep the first r columns corresponding to the first r singular values
    V = vr[:,:r]

    #Calculate U
    #The first r columns are 1/sigma*A V_i where V_i is the ith column of V.
    #Only use columns that correspond to the first r singular values
    U = 1./sigs[:r]*A.dot(V[:,:r])
    
    return U, s, V.conj().T

def svd_approx(A, k):
    '''
    Calculate the best rank k approximation to A with respect to the induced
    2-norm. Use the SVD.
    Inputs:
        A -- array of shape (m,n)
        k -- positive integer
    Returns:
        Ahat -- best rank k approximation to A obtained via the SVD
    '''
    #compute the reduced SVD
    U,s,Vh = svd(A,full_matrices=False)
    
    #keep only the first k singular values
    S = np.diag(s[:k])
    
    #reconstruct the best rank k approximation
    return U[:,:k].dot(S).dot(Vt[:k,:])
    
def plot_svd():
	A = np.array([[3,1],[1,3]])
	U, S, V = truncated_svd(A)
	S = np.diag(S)
	
	t = np.linspace(0,2*np.pi,100)
	pts = np.array([np.cos(t),np.sin(t)])
	v= V.dot(pts)
	sv = S.dot(v)
	a = U.dot(sv)
	
	unit_vecs = np.array([[1,0],[0,1]])
	vu = V.dot(unit_vecs)
	svu = S.dot(vu)
	au = U.dot(svu)
	
	plt.subplot(221)
	plt.plot(pts[0],pts[1],'b')
	plt.plot([0,unit_vecs[0,0]],[0,unit_vecs[1,0]],'g')
	plt.plot([0,unit_vecs[0,1]],[0,unit_vecs[1,1]],'g')
	plt.axis('equal')

	plt.subplot(222)
	plt.plot(v[0],v[1],'b')
	plt.plot([0,vu[0,0]],[0,vu[1,0]],'g')
	plt.plot([0,vu[0,1]],[0,vu[1,1]],'g')
	plt.axis('equal')

	plt.subplot(223)
	plt.plot(sv[0],sv[1],'b')
	plt.plot([0,svu[0,0]],[0,svu[1,0]],'g')
	plt.plot([0,svu[0,1]],[0,svu[1,1]],'g')
	plt.axis('equal')
	
	plt.subplot(224)
	plt.plot(a[0],a[1],'b')
	plt.plot([0,au[0,0]],[0,au[1,0]],'g')
	plt.plot([0,au[0,1]],[0,au[1,1]],'g')
	plt.axis('equal')
	plt.show()
    
def lowest_rank_approx(A,e):
    '''
    Calculate the lowest rank approximation to A that has error strictly less than e.
    Inputs:
        A -- array of shape (m,n)
        e -- positive floating point number
    Returns:
        Ahat -- the best rank s approximation of A constrained to have error less than e, 
                where s is as small as possible.
    '''
    #calculate the reduced SVD
    U,s,Vh = svd(A,full_matrices=False)
    
    #find the index of the first singular value less than e
    k = np.where(s<e)[0][0] 
    
    #now recreate the rank k approximation
    S = np.diag(s[:k])
    return U[:,:k].dot(S).dot(Vt[:k,:])
    

def readimg(filename, channel=None):
    if channel is not None:
        return sp.misc.imread(filename)[:,:,channel]
    else:
        return sp.misc.imread(filename)


def compressSVD(filename, rank, random=False, channel=None):
    img = readimg(filename, channel)

    try:
        isize = img[:,:,0].shape
        colors = [la.svd(img[:,:,i]) for i in range(3)]
    except IndexError:
        isize = img.shape
        plt.gray()
        colors = la.svd(img)

    plt.ion()
    imgc = plt.imshow(img)
    newimg = sp.zeros_like(img)

    rank = range(1,rank+1)

    if random is True:
        sp.random.shuffle(rank)

    for r in rank:
        col_res = hat(colors, r-1, r)
        try:
            #col_res[0] = sp.where(col_res[0]>255, col_res[0], 255)
            #col_res[1] = sp.where(col_res[1]>255, col_res[1], 255)
            #col_res[2] = sp.where(col_res[2]>255, col_res[2], 255)

            #col_res[0] = sp.where(col_res[0]<0, col_res[0], 0)
            #col_res[1] = sp.where(col_res[1]<0, col_res[1], 0)
            #col_res[2] = sp.where(col_res[2]<0, col_res[2], 0)
            
            newimg[:,:,0] += col_res[0]
            newimg[:,:,1] += col_res[1]
            newimg[:,:,2] += col_res[2]

            ## for ch in range(3):
            ##     newimg[newimg[:,:,ch]<1]=0
            ##     newimg[newimg[:,:,ch]>254]=255
                
        except IndexError:
            newimg += col_res[0]
            ## newimg[newimg<1]=0
            ## newimg[newimg>254]=255
        
        imgc.set_data(newimg)
        plt.draw()
    plt.ioff()
    plt.show()

    return newimg


def hat(color_svd, lrank, urank):
    results = []
    if len(color_svd) == 3:
        r = 3
    else:
        r = 1
        
    for c in range(r):
        U = color_svd[c][0]
        S = sp.diag(color_svd[c][1])
        Vt = color_svd[c][2]
        results.append(U[:,lrank:urank].dot(S[lrank:urank, lrank:urank]).dot(Vt[lrank:urank,:]))
        
    return results
    
def matrix_rank(X):
    """Compute the rank of a matrix using the SVD"""
    
    S = la.svd(X, compute_uv=False)
    tol = S.max()*sp.finfo(S.dtype).eps
    return sum(S>tol)
