#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 01 09:25:55 2025

@author: jordiromero




SIMULATION CODE STARTS AT LINE 2550
"""
import os
import time
import numpy as np
import scipy as sp
import picos as pc

#from scipy.optimize import least_squares
#import cvxpy as cp
#import numpy as np
#import matplotlib.pyplot as plt
#from itertools import permutations
from itertools import combinations
import itertools
#import cvxpy as cp
#import numpy as np
#import matplotlib.pyplot as plt

#import sys
#import numpy as np
import math
# import scipy.io
from datetime import datetime
from os.path import exists
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 17:11:42 2025

@author: jordiromero
"""


def basis_element(dim, i):
    """
    Create a basis vector in a complex Hilbert space of dimension 'dim' with 1 in the i-th position.

    Args:
    dim (int): The dimension of the basis.
    i (int): The position of the nonzero entry in the basis vector.
    
    Returns:
    torch.Tensor: The i-th basis vector with shape (dim, 1).
    """
    keti = torch.zeros((dim, 1)).type(torch.complex128).to(device)
    keti[i] = 1
    return keti

def id(dim):
    """
    Create an identity matrix of dimension 'dim'.

    Args:
    dim (int): The dimension of the identity matrix.
    
    Returns:
    torch.Tensor: The identity matrix of dimension 'dim'.
    """
    return (torch.eye(dim) + 0j).type(torch.complex128)

def random_unitary(dim):
    """
    Create a random unitary matrix of dimension 'dim'.

    Args:
    dim (int): The dimension of the unitary matrix.
    
    Returns:
    torch.Tensor: The random unitary matrix of dimension 'dim'.
    """
    m = (torch.randn(dim, dim) + 1j * torch.randn(dim, dim)).type(torch.complex128).to(device)
    q, r = torch.linalg.qr(m)
    return q @ torch.diag_embed(torch.diag(r) / torch.abs(torch.diag(r)))

def rand_rho_pure(dim1, dim2):
    """
    Create a random pure density matrix for a bipartite system.

    Args:
    dim1, dim2 (int): The dimensions of the two subsystems.
    
    Returns:
    torch.Tensor: The random pure density matrix.
    """
    dim = dim1 * dim2
    psi = basis_element(dim, 0)
    u = random_unitary(dim)
    psi = u @ psi
    return psi @ psi.conj().T

def rand_rho_purification(dim1, dim2):
    """
    Create a random purified density matrix for a bipartite system.

    Args:
    dim1, dim2 (int): The dimensions of the two subsystems.
    
    Returns:
    torch.Tensor: The random purified density matrix.
    """
    dim = dim1 * dim2
    psi = basis_element(dim ** 2, 0)
    u = random_unitary(dim ** 2)
    psi = u @ psi
    return psi @ psi.conj().T

def partial_trace_kraus_operator(dim1,dim2, k):
    """
    Create the k-th Kraus operator for the partial trace.

    Args:
    dim1, dim2 (int): The dimensions of the two subsystems.
    k (int): The index of the Kraus operator.
    
    Returns:
    torch.Tensor: The k-th Kraus operator.
    """
    id = torch.eye(dim1).type(torch.complex128).to(device)
    Ek = torch.kron(id,basis_element(dim2, k))
    return Ek

def partial_trace_kraus_operators(dim1,dim2):
    """
    Create all Kraus operators for the partial trace.

    Args:
    dim1, dim2 (int): The dimensions of the two subsystems.
    
    Returns:
    list[torch.Tensor]: The list of all Kraus operators.
    """
    return [partial_trace_kraus_operator(dim1,dim2, k) for k in range(dim2)]

def partial_trace(rho, dim1, kraus_operators):
    """
    Perform the partial trace of a density matrix over the second system.

    Args:
    rho (torch.Tensor): The density matrix.
    dim1 (int): The dimension of the subsystem to keep.
    kraus_operators (list[torch.Tensor]): The Kraus operators for the partial trace.
    
    Returns:
    torch.Tensor: The partial trace of the density matrix.
    """
    pt = torch.zeros((dim1, dim1)).type(torch.complex128).to(device)
    for Ek in kraus_operators:
        #print(np.shape(Ek),np.shape(rho))
        #print(rho)
        #print(Ek)
        pt += Ek.conj().T @ rho @ Ek
    return pt

def partial_transpose(matrix, d1, d2, system=2):
    """
    Perform the partial transpose of a matrix.

    Args:
    matrix (torch.Tensor): The matrix to transpose.
    d1, d2 (int): The dimensions of the two subsystems.
    system (int): The index of the subsystem to transpose (1 or 2).
    
    Returns:
    torch.Tensor: The partial transpose of the matrix.
    """
    tensor = matrix.reshape(d1, d2, d1, d2)
    if system == 1:
        transposed_tensor = torch.permute(tensor, (2, 1, 0, 3))
    elif system == 2:
        transposed_tensor = torch.permute(tensor, (0, 3, 2, 1))
    else:
        raise ValueError("Invalid value for system. Must be 1 or 2.")
    transposed_matrix = transposed_tensor.reshape(d1 * d2, d1 * d2)
    return transposed_matrix

def optimal_witness3x3(alpha):
    a=2/3*(1+1/2*np.cos(alpha)+np.sqrt(3)/2*np.sin(alpha))
    b=2/3*(1-np.cos(alpha))
    c=2/3*(1+1/2*np.cos(alpha)-np.sqrt(3)/2*np.sin(alpha))
    
    Wit= 1/6*np.array([
        [a,  0, 0,   0,  -1,  0,   0,  0,  -1],
        [0,  b, 0,   0,  0,  0,   0,  0,  0],
        [0,   0,  c,   0,  0,  0,   0,  0,  0],
        [0,   0,  0,   b,  0,  0,   0,  0,  0],
        [-1,   0,  0,   0, c, 0,   0,  0,  -1],
        [0,   0,  0,   0, 0, a,   0,  0,  0],
        [0,   0,  0,   0,  0,  0,   c,  0,  0],
        [0,   0,  0,   0,  0,  0,   0,  a,  0],
        [-1,   0,  0,   0,  -1,  0,  0,  0,  b]
    ], dtype=complex)

    return Wit



def rand_rho(dim1, dim2, kraus=None):
    """
    Generate a random density matrix for a bipartite system using purification.

    Args:
    dim1, dim2 (int): The dimensions of the two subsystems.
    kraus (list[torch.Tensor], optional): The Kraus operators for the partial trace. 
    
    Returns:
    torch.Tensor: The random density matrix.
    """
    rho_pure = rand_rho_purification(dim1, dim2)
    if kraus is None:
        kraus = partial_trace_kraus_operators(dim1 * dim2, dim1 * dim2)
    return partial_trace(rho_pure, dim1 * dim2, kraus)

def random_functional(dim1, dim2):
    """
    Generate a random Hermitian matrix (functional) with unit trace.

    Args:
    dim1, dim2 (int): The dimensions of the two subsystems.
    
    Returns:
    torch.Tensor: The random Hermitian matrix with unit trace.
    """
    m = (torch.randn((dim1 * dim2, dim1 * dim2)).type(torch.complex128) + 1j * torch.randn((dim1 * dim2, dim1 * dim2)).type(torch.complex128)).to(device)
    m = m + m.conj().T
    return m / (torch.trace(m @ m.conj().T) ** 0.5)

def random_witness_from_partial_transpose(dim1, dim2):
    """
    Generate a random entanglement witness from the partial transpose of a pure state.

    Args:
    dim1, dim2 (int): The dimensions of the two subsystems.
    
    Returns:
    torch.Tensor: The random entanglement witness.
    """
    pure_rho = rand_rho_pure(dim1, dim2)
    return partial_transpose(pure_rho, dim1, dim2, system=2)

def random_witness_from_family(dim1, dim2):
    """
    Generate a random entanglement witness from a family of witnesses for qubits.

    Args:
    dim1, dim2 (int): The dimensions of the two subsystems.

    Returns:
    torch.Tensor: The random entanglement witness.

    Raises:
    AssertionError: If dimensions aren't equal to 2 (the function is implemented only for qubits).
    """
    assert dim1 == dim2 and dim1 == 2, "Witness from family only implemented for qubits."
    
    w = torch.zeros((dim1 * dim2, dim1 * dim2)).type(torch.complex128).to(device)
    params = 2 * torch.rand(3) - 1
    alpha, beta, gamma = params[0], params[1], params[2]
    gamma = gamma * torch.sqrt((alpha**2 + beta**2)/2)
    
    alpha = alpha.type(torch.complex128)
    beta = beta.type(torch.complex128)
    gamma = gamma.type(torch.complex128)
    w[0,0] = 1 + gamma
    w[1,1] = 1 - gamma
    w[2,2] = 1 - gamma
    w[3,3] = 1 + gamma
    w[3,0] = alpha + beta
    w[0,3] = alpha + beta
    w[1,2] = alpha - beta
    w[2,1] = alpha - beta
    return w / 4

def random_optimal3x3witness(dim1,dim2):
    alpha=np.random.uniform(np.pi/3,5/(3*np.pi))
    return torch.from_numpy(optimal_witness3x3(alpha))


def is_positive_semidefinite(matrix):
    """
    Checks if the given matrix is positive semidefinite.
    
    Args:
    matrix (torch.Tensor): The matrix to check.
    
    Returns:
    bool: True if the matrix is positive semidefinite, False otherwise.
    """
    if not torch.allclose(matrix, matrix.conj().T):
        return False

    min_eig = torch.linalg.eigvalsh(matrix).min()
    return min_eig >= 0 or torch.isclose(min_eig, torch.tensor(0.0).type(min_eig.dtype))


def is_entangled_ppt(rho, dim1, dim2):
    """
    Checks if the given density matrix is entangled using the Peres-Horodecki criterion (PPT).
    
    Args:
    rho (torch.Tensor): The density matrix to check.
    dim1 (int): The dimension of the first subsystem.
    dim2 (int): The dimension of the second subsystem.
    
    Returns:
    bool: True if the matrix is entangled, False otherwise.
    """
    rho_t = partial_transpose(rho, dim1, dim2, system=2)
    return not is_positive_semidefinite(rho_t)

#Enhanced Realignment Criterion

def Realignment_operator(rho,dim1,dim2):
    """
    Realignment (reshuffling) of a bipartite density matrix.

    Parameters:
    - rho: the input density matrix (as a 2D NumPy array)
    - dimA: dimension of subsystem A
    - dimB: dimension of subsystem B

    Returns:
    - realigned matrix (as a NumPy array)
    """
    rho = rho.reshape((dim1, dim2, dim1, dim2))
    # Swap axes to realign: (i, j, k, l) -> (i, k, j, l)
    rho = np.transpose(rho, (0, 2, 1, 3))
    # Flatten into 2D matrix
    return rho.reshape((dim1 * dim1, dim2 * dim2))
    
    
    
    
    
    
def CCNR_criterion(rho,dim1,dim2):
    llista=partial_trace_kraus_operators(dim1, dim2)
    U=build_permutation_unitary(dim1, 2, 0, 1)
    U=torch.from_numpy(U)
    rhoa=partial_trace(rho,dim1,llista)
    rhob=partial_trace(U@rho@np.transpose(U), dim1, llista)
    rhof=rho-np.kron(rhoa,rhob)
    #rhoa=rhoa.resolve_conj().numpy()
    #rhob=rhob.resolve_conj().numpy()
    ub=(np.sqrt(1-np.trace(rhoa@rhoa))*np.sqrt(1-np.trace(rhob@rhob))).real
    rmatrix=Realignment_operator(rhof, dim1, dim2)
    rmatrix0=Realignment_operator(rho, dim1, dim2)
    r=np.trace(sp.linalg.sqrtm(rmatrix@rmatrix.conj().T)).real
    r0=np.trace(sp.linalg.sqrtm(rmatrix0@rmatrix0.conj().T)).real
    print(r,ub,r0)
    if r>ub or r0>1:
        return True #matrix is entangled
    else:
        return False #no idea
 
    

def build_permutation_unitary(d, N, i, j):
    """
    Create a unitary matrix that permutes indices i and j
    in the computational basis of a d^N-dimensional Hilbert space.

    Args:
        d (int): Local dimension (e.g. 2 for qubits, 3 for qutrits)
        N (int): Number of parties
        i (int): First index to permute
        j (int): Second index to permute

    Returns:
        numpy.ndarray: A (d^N, d^N) unitary matrix representing the permutation
    """
    dim = d ** N
    U = np.zeros((dim, dim), dtype=complex)

    # Map from tuple -> index
    def state_to_index(state):
        idx = 0
        for k in range(N):
            idx = idx * d + state[k]
        return idx

    # Loop over all basis states
    for state in itertools.product(range(d), repeat=N):
        type(state)
        permuted = list(state)
        permuted[i], permuted[j] = permuted[j], permuted[i]

        from_idx = state_to_index(state)
        to_idx = state_to_index(permuted)

        U[to_idx, from_idx] = 1.0  # Acts like a permutation matrix

    return U




def split_combinations(arr, a):
    if not (0 <= a <= len(arr)):
        raise ValueError("Parameter 'a' must be between 0 and the length of the array.")
    
    result = []
    for combo in combinations(range(len(arr)), a):
        group1 = [arr[i] for i in combo]
        group2 = [arr[i] for i in range(len(arr)) if i not in combo]
        result.append((group1, group2))
    
    return result




def array_to_python_int_tuple(arr):
    return tuple(int(x) for x in arr)






def Symm_Extension(rho,dim1,dim2,layers): 
    """
    Checks if the given density matrix is entangled by checking whether there exists a symmetric extension (DPS hierarchy).
    Args:
    rho (torch.Tensor): The density matrix to check.
    dim1 (int): The dimension of the first subsystem.
    dim2 (int): The dimension of the second subsystem.
    layers: how many layers of the hierarchy are we looking at (takes by default that our initial state is bipartite) at most 5 for the moment (3 layers is already not immediate and 4 layers takes quite a bit of time )
    Returns:
    bool: True if the matrix is entangled, False otherwise.    
    """
    rho1 = rho.numpy()
    rho1=rho1/np.trace(rho1)
    #print(np.trace(rho1), np.trace(rho))
    dims = (dim1,dim1,dim1)
    dims1 = (dim1,dim1,dim1,dim1)
    dims2 = (dim1,dim1,dim1,dim1,dim1)
    dims3 = (dim1,dim1,dim1,dim1,dim1,dim1)
    dims4 = (dim1,dim1,dim1,dim1,dim1,dim1,dim1)
    dimss = [dims,dims1,dims2,dims3,dims4]
    traces = [2]
    parties = [0,2]
    nparties = [0,1]
    for i in range(layers):
        start = time.time()
        print('layer #',i+1)
        N = 3+i
        nparties = np.concatenate((nparties,np.array([2+i])),axis=0)
        
        
        # then, the constraints: postiivity and marginals
        F = pc.Problem()
        sigma = pc.HermitianVariable('sigma',dim1**N )
        #x = pc.RealVariable('x')
        prob = sigma.tr
        F.set_objective("min",prob)
        F.add_constraint(sigma>>0)#-x*id(dim1**N).numpy() >> 0)
        F.add_constraint(np.real(sigma.tr) == 1)
        
        if i==0: # the partial trace of all added parties recovers the initial state
            U=build_permutation_unitary(dim1, N, 1,2 )
            F.add_constraint(sigma.partial_trace((2), dims) == rho1)
            F.add_constraint( U*sigma*U.T== sigma)
            
        elif i==1:
            U = build_permutation_unitary(dim1, N, 1,2 )
            U1 = build_permutation_unitary(dim1, N, 1,3 )
            F.add_constraint(sigma.partial_trace((2,3),dims1) == rho1)
            F.add_constraint( U*sigma*U.T== sigma)
            F.add_constraint( U1*sigma*U1.T== sigma)
        elif i==2:
            U = build_permutation_unitary(dim1, N, 1,2 )
            U1 = build_permutation_unitary(dim1, N, 1,3 )
            U2 = build_permutation_unitary(dim1, N, 1,4 )
            F.add_constraint( U*sigma*U.T== sigma)
            F.add_constraint( U1*sigma*U1.T== sigma)
            F.add_constraint( U2*sigma*U2.T== sigma)
            F.add_constraint(sigma.partial_trace((2,3,4),dims2) == rho1)
        elif i==3:
            U = build_permutation_unitary(dim1, N, 1,2 )
            U1 = build_permutation_unitary(dim1, N, 1,3 )
            U2 = build_permutation_unitary(dim1, N, 1,4 )
            U3 = build_permutation_unitary(dim1, N, 1,5 )
            F.add_constraint( U*sigma*U.T== sigma)
            F.add_constraint( U1*sigma*U1.T== sigma)
            F.add_constraint( U2*sigma*U2.T== sigma)
            F.add_constraint( U3*sigma*U3.T== sigma)
            F.add_constraint(sigma.partial_trace((2,3,4,5),dims3) == rho1)
        elif i==4:
            U = build_permutation_unitary(dim1, N, 1,2 )
            U1 = build_permutation_unitary(dim1, N, 1,3 )
            U2 = build_permutation_unitary(dim1, N, 1,4 )
            U3 = build_permutation_unitary(dim1, N, 1,5 )
            U4 = build_permutation_unitary(dim1, N, 1,6 )
            F.add_constraint( U*sigma*U.T== sigma)
            F.add_constraint( U1*sigma*U1.T== sigma)
            F.add_constraint( U2*sigma*U2.T== sigma)
            F.add_constraint( U3*sigma*U3.T== sigma)
            F.add_constraint( U4*sigma*U4.T== sigma)
            F.add_constraint(sigma.partial_trace((2,3,4,5,6),dims4) == rho1)
        
        for j in range(3+i):
            F.add_constraint(sigma.partial_transpose((j), dimss[N-3])>>0) # all partial transpose of 1 party is semidefinite positive
        
        
        if N>=4:
            pt=split_combinations(nparties,2)
            #print(pt)
            for k in range(int(len(pt)/2)):
                F.add_constraint(sigma.partial_transpose(array_to_python_int_tuple(pt[k][0]),dimss[N-3] )>>0) # all partial transpose of 2 parties is semidefinite positive       
        if N>=6:
            pt2=split_combinations(nparties,3)
            
            for l in range(int(len(pt2)/2)):
                F.add_constraint(sigma.partial_transpose(array_to_python_int_tuple(pt[l][0]),dimss[N-3] )>>0) # all partial transpose of 3 parties is semidefinite positive
        
        #dims = np.concatenate((dims, np.array([dim1])), axis=0)
        traces= np.concatenate((traces, np.array([i+3])), axis=0)
        parties=np.concatenate((parties, np.array([i+3])), axis=0)
        #constraints += [generic_permutation(sigma,dims,parties)==sigma]
        
        # we minimize the trace of sigma
        
        
        # solve the problem
        print('solvingtime')
        F.solve(solver = 'mosek', primals = False, verbosity = False)
        
        end=time.time()
        print("Status = ", F.status, 'Time = ', (end-start)//3600,'h',(((end-start)/3600)-(end-start)//3600)*3600//60,'min',((((end-start)/3600)-(end-start)//3600)*3600/60-(((end-start)/3600)-(end-start)//3600)*3600//60)*60,'s')
        #print(sigma.partial_trace((2),dims)-rho1)
        #print(prob.value)
        if F.status != 'optimal':
            return True  #it has not found a ppt extension aka is entangled
        elif F.status=='optimal' and i==layers-1:
            return False #it has found a ppt extension for all the checked layers aka no idea
        


def mutual_information_values_ent(values_vector, entangled_vector):
    """
    Calculates the mutual information between the values vector and the entangled vector.
    
    Args:
    values_vector (torch.Tensor): The vector of values.
    entangled_vector (torch.Tensor): The vector of entanglement labels.
    
    Returns:
    torch.Tensor: The mutual information.
    """
    events = torch.stack((values_vector, entangled_vector), dim=1)
    counts, _ = torch.histogramdd(events, bins=[100, 2])
    
    pJoint = counts / counts.sum()
    pValue = pJoint.sum(dim=1)
    pEntangled = pJoint.sum(dim=0)
    
    hValue = - torch.special.xlogy(pValue, pValue).sum()
    hEntangled = - torch.special.xlogy(pEntangled, pEntangled).sum()
    hJoint = - torch.special.xlogy(pJoint, pJoint).sum()

    return hValue + hEntangled - hJoint


def mutual_information_sign_ent(values_vector, entangled_vector):
    """
    Calculates the mutual information between the signs of the values vector and the entangled vector.
    
    Args:
    values_vector (torch.Tensor): The vector of values.
    entangled_vector (torch.Tensor): The vector of entanglement measures.
    
    Returns:
    torch.Tensor: The mutual information.
    """
    signs_vector = torch.sign(values_vector)
    events = torch.stack((signs_vector, entangled_vector), dim=1)
    counts, _ = torch.histogramdd(events, bins=[2, 2])
    
    pJoint = counts / counts.sum()
    pSign = pJoint.sum(dim=1)
    pEntangled = pJoint.sum(dim=0)
    
    hSign = - torch.special.xlogy(pSign, pSign).sum()
    hEntangled = - torch.special.xlogy(pEntangled, pEntangled).sum()
    hJoint = - torch.special.xlogy(pJoint, pJoint).sum()

    return hSign + hEntangled - hJoint

#####

# Matrix Functions:
# Outer product of two vectors
def Outer(p1, p2):
    return (np.outer(np.array(p1), np.array(p2)).flatten())


# Normalization of a vector
def Normalize(p):
    p2 = np.dot(p, np.conj(p))
    p2 = math.sqrt(np.real(p2))
    return (p / p2)

#Normalization of a pure state
def NormPure(p):
    p2=1/math.sqrt(np.dot(p,np.conj(p)))
    return(np.multiply(p,p2))

# Build a projection from a vector
def Project(p1):
    return (np.outer(np.array(p1), np.conj(np.array(p1))))


# Scalar product of two matrices
def Product(t1, t2):
    #k=np.trace(np.matmul(t1,t2))
    #return(np.real(k))
    t1a = np.ravel(t1, order='C')
    t2a = np.ravel(t2, order='F')
    return (np.real(np.dot(t1a, t2a)))


# mean value for a vector
def Product1(m1, v1):
    vvv1 = np.transpose([v1])
    vvv2 = np.conjugate([v1])
    return (np.dot(vvv2, np.dot(m1, v1))[0])


# Generate a random vector with Haar measure
def Generate(d):
    k1 = np.random.normal(0, 1, d)
    k2 = np.random.normal(0, 1, d)
    k3 = k1 + complex(0, 1) * k2
    return (k3)

# Generate a random state in UPB
#rho4: list of UPB states
def RandomUPB(rho4):
    psi1=Normalize(Generate(len(rho4)))
    return Project(np.dot(psi1,rho4))


# Identity Matrix
def IdMatrix(d):
    p1 = np.eye(int(d), dtype=complex)
    return (p1)


# Kronecker Product
def Kronecker(p1, p2):
    return (np.kron(p1, p2))


# Expand an operator to n qubits
def Expand2FS(p1, n, j1):
    return (Kronecker(Kronecker(np.eye(2 ** j1), p1), np.eye(2 ** (n - j1 - 1))))


# Expand an operator to n quDits
def ExpanddFS(p1, d, n, j1):
    return (Kronecker(Kronecker(np.eye(int(d ** j1)), p1), np.eye(int(d ** (n - j1 - 1)))))


# sandwich an operator with a unitarry
def Rotate(rho2, U):
    rho2a = np.matmul(rho2, np.conj(U).T)
    rho2a = np.matmul(U, rho2a)
    return rho2a


# MTX file read and write
# [1]: 'matrix' 'vector'
# [2]: 'array' 'coordinate'
# [3]: 'integer' 'real' 'complex' 'pattern'
# [4]: 'general' 'symmetric' 'skew-symmetric' 'hermitian

def getnum(file, vartype):
    readnum = file.readline()
    if vartype == 0:
        return (int(readnum))
    elif vartype == 1:
        return (float(readnum))
    elif vartype == 2:
        readnum = readnum.split()
        kreadnum = list(map(float, readnum))
        return (kreadnum[0] + complex(0, 1) * kreadnum[1])


def get3num(file, vartype):
    readnum = file.readline()
    readnum = readnum.split
    if vartype == 0:
        return ([int(readnum[0]), int(readnum[1]), int(readnum[2])])
    elif vartype == 1:
        return ([int(readnum[0]), int(readnum[1]), float(readnum[2])])
    elif vartype == 2:
        return ([int(readnum[0]), int(readnum[1]), float(readnum[2]) + complex(float(readnum[3]))])


def readmtx(filename):
    with open(filename, "r") as file:
        line = str(file.readline())
        firstline = line.split()
        if firstline[1] == 'matrix':
            shape = 0
        elif firstline[1] == 'vector':
            shape = 1
        if firstline[2] == 'array':
            descr = 0
        elif firstline[2] == 'coordinate':
            descr = 1
        if firstline[3] == 'integer':
            vartype = 0
        elif firstline[3] == 'real':
            vartype = 1
        elif firstline[3] == 'complex':
            vartype = 2
        if firstline[4] == 'general':
            symtype = 0
        elif firstline[4] == 'symmetric':
            symtype = 1
        elif firstline[4] == 'skew-symmetric':
            symtype = 2
        elif firstline[4] == 'hermitian':
            symtype = 3
        while line[0] == "%" or len(line) == 1:
            line = file.readline()
        line = list(map(int, line.split()))
        if shape == 0 and descr == 0:
            rows = line[0]
            cols = line[1]
            if vartype == 0:
                wynik = np.zeros((rows, cols), dtype=int)
            elif vartype == 1:
                wynik = np.zeros((rows, cols), dtype=float)
            elif vartype == 2:
                wynik = np.zeros((rows, cols), dtype=complex)
            if symtype == 0:
                for i1 in range(cols):
                    for i2 in range(rows):
                        wynik[i2][i1] = getnum(file, vartype)
            elif symtype == 1 and rows == cols:
                for i1 in range(cols):
                    for i2 in range(i1, rows):
                        wynik[i2][i1] = getnum(file, vartype)
                        wynik[i1][i2] = wynik[i2][i1]
            elif symtype == 2 and rows == cols:
                for i1 in range(0, cols):
                    for i2 in range(i1 + 1, rows):
                        wynik[i2][i1] = getnum(file, vartype)
                        wynik[i1][i2] = -wynik[i2][i1]
            elif symtype == 3 and rows == cols:
                for i1 in range(0, cols):
                    for i2 in range(i1, rows):
                        wynik[i2][i1] = getnum(file, vartype)
                        wynik[i1][i2] = np.conj(wynik[i2][i1])
        if shape == 0 and descr == 1:
            rows = line[0]
            cols = line[1]
            nonzeros = line[3]
            if vartype == 0:
                wynik = np.zeros((cols, rows), dtype=int)
            elif vartype == 1:
                wynik = np.zeros((cols, rows), dtype=float)
            elif vartype == 2:
                wynik = np.zeros((cols, rows), dtype=complex)
            for i1 in range(nonzeros):
                entry = get3num(file, vartype)
                wynik[entry[1]][entry[0]] = entry[2]
            if symtype == 1 and rows == cols:
                for i1 in range(rows):
                    for i2 in range(i1, cols):
                        if np.abs(wynik[i1][i2]) != 0:
                            wynik[i2][i1] = wynik[i1][i2]
                        else:
                            wynik[i1][i2] = wynik[i2][i1]
            elif symtype == 2 and rows == cols:
                for i1 in range(0, cols):
                    for i2 in range(i1 + 1, rows):
                        if np.abs(wynik[i1][i2]) != 0:
                            wynik[i2][i1] = -wynik[i1][i2]
                        else:
                            wynik[i1][i2] = -wynik[i2][i1]
            elif symtype == 3 and rows == cols:
                for i1 in range(0, cols):
                    for i2 in range(i1, rows):
                        if np.abs(wynik[i1][i2]) != 0:
                            wynik[i2][i1] = np.conj(wynik[i1][i2])
                        else:
                            wynik[i1][i2] = np.conj(wynik[i2][i1])
        file.close()
    return (wynik)


def writemtx(filename, lista, vartype):
    if vartype == 1:
        vartype1 = "real"
    elif vartype == 2:
        vartype1 = "complex"
    elif vartype == 0:
        vartype1 = "integer"
    with open(filename, 'w') as file:
        file.write(" ".join(["%%MatrixMarket matrix array", vartype1, "general\n"]))
        file.write("%Generated by CSSFinder\n")
        file.write(str(len(lista)))
        file.write("  ")
        file.write(str(len(lista[0])))
        file.write("\n")
        for i1 in range(len(lista[0])):
            for i2 in range(len(lista)):
                if vartype == 2:
                    file.write(str(np.real(lista[i2][i1])))
                    file.write("   ")
                    file.write(str(np.imag(lista[i2][i1])))
                    file.write("\n")
                else:
                    file.write(str(lista[i2][i1]))
                    file.write("\n")
        file.close()


# Random states:
# n qubit state
def Random2FS(n):
    q1 = Normalize(Generate(2))
    if n > 1:
        for l1 in range(n - 1):
            q1 = Outer(q1, Normalize(Generate(2)))
    return (Project(q1))


# n quDit state
def RandomdFS(d, n):
    q1 = Normalize(Generate(d))
    for l1 in range(n - 1):
        q1 = Outer(q1, Normalize(Generate(d)))
    return (Project(q1))


# biseparable state
def RandomBS(d1, d2):
    return Project(Outer(Normalize(Generate(d1)), Normalize(Generate(d2))))


# biseparable state with three quDits
def Random3P(d1, swaps, i):
    if i == 0:
        # aBC
        return (RandomBS(d1, d1 * d1))
    if i == 1:
        # AbC
        return (Rotate(RandomBS(d1, d1 * d1), swaps[0]))
    if i == 2:
        # ABc
        return (RandomBS(d1 * d1, d1))


# biseparable for  4 quDits
def Random4P(d1, swaps, i):
    if i == 0:
        # aBCD
        return (RandomBS(d1, d1 * d1 * d1))
    if i == 1:
        # AbCD
        return (Rotate(RandomBS(d1, d1 * d1 * d1), swaps[0]))
    if i == 2:
        # ABcD
        return (Rotate(RandomBS(d1 * d1 * d1, d1), swaps[3]))
    if i == 3:
        # ABCd
        return (RandomBS(d1 * d1 * d1, d1))
    if i == 4:
        # abCD
        return (RandomBS(d1 * d1, d1 * d1))
    if i == 5:
        # aBcD
        return (Rotate(RandomBS(d1 * d1, d1 * d1), swaps[2]))
    if i == 6:
        # aBCd
        return (Rotate(RandomBS(d1 * d1, d1 * d1), swaps[1]))


def Random4k3(d1, swaps, i):
    if i == 0:
        # aBCD
        return (RandomBS(d1, d1 * d1 * d1))
    if i == 1:
        # AbCD
        return (Rotate(RandomBS(d1, d1 * d1 * d1), swaps[0]))
    if i == 2:
        # ABcD
        return (Rotate(RandomBS(d1 * d1 * d1, d1), swaps[1]))
    if i == 3:
        # ABCd
        return (RandomBS(d1 * d1 * d1, d1))


def Random5k4(d1, swaps, i):
    if i == 0:
        # aBCDE
        return (RandomBS(d1, d1 ** 4))
    if i == 1:
        # AbCDE
        return (Rotate(RandomBS(d1, d1 ** 4), swaps[0]))
    if i == 2:
        # ABcDE
        return (Rotate(RandomBS(d1, d1 ** 4), swaps[1]))
    if i == 3:
        # ABCdE
        return (Rotate(RandomBS(d1 ** 4, d1), swaps[2]))
    if i == 4:
        # ABCDe
        return (RandomBS(d1 ** 4, d1))


def Random6k5(d1, swaps, i):
    if i == 0:
        # aBCDEF
        return (RandomBS(d1, d1 ** 5))
    if i == 1:
        # AbCDEF
        return (Rotate(RandomBS(d1, d1 ** 5), swaps[0]))
    if i == 2:
        # ABcDEF
        return (Rotate(RandomBS(d1, d1 ** 5), swaps[1]))
    if i == 3:
        # ABCdEF
        return (Rotate(RandomBS(d1 ** 5, d1), swaps[2]))
    if i == 4:
        # ABCdEF
        return (Rotate(RandomBS(d1 ** 5, d1), swaps[5]))
    if i == 4:
        # ABCDEf
        return (RandomBS(d1 ** 5, d1))


# Rendom Unitaries
# Biseparability
def RandomUBS(a, d1, d2):
    if a == 0:
        rubsp1 = (math.cos(0.01 * math.pi) + complex(0, 1) * math.sin(0.01 * math.pi) - 1) * Project(
            Normalize(Generate(d1))) + np.eye(d1)
        return (Kronecker(rubsp1, np.eye(int(d2))))
    if a == 1:
        rubsp1 = (math.cos(0.01 * math.pi) + complex(0, 1) * math.sin(0.01 * math.pi) - 1) * Project(
            Normalize(Generate(d2))) + np.eye(d2)
        return (Kronecker(np.eye(int(d1)), rubsp1))
        # return(Kronecker(IdMatrix(d1),unitatry_group.rvs(d2)))


# n qubits    
def RandomU2FS(n, j):
    # p1=unitary_group.rvs(2)
    p1 = (math.cos(0.01 * math.pi) + complex(0, 1) * math.sin(0.01 * math.pi) - 1) * Random2FS(1) + np.eye(2)
    return (Expand2FS(p1, n, j))


# n quDits
def RandomUdFS(d, n, j):
    # p1=unitary_group.rvs(d)
    p1 = (math.cos(0.01 * math.pi) + complex(0, 1) * math.sin(0.01 * math.pi) - 1) * RandomdFS(d, 1) + np.eye(d)
    return (ExpanddFS(p1, d, n, j))

# UPB three qubits
def RandomUUPB(rho4):
    p0=len(rho4)
    p1 = Normalize(Generate(p0))
    p2 =np.dot(p1,rho4)
    return (math.cos(0.01*math.pi)+1j*math.sin(0.01*math.pi)-1)*Project(p2)+np.eye(27)

# Optimizers
# biseparability
def OptimizeBS(rho2, rho3, d1, d2):
    pp1 = Product(rho2, rho3)
    psi1 = Normalize(np.conjugate(rho2[1]))  #
    for obsj1 in range(200):
        U = RandomUBS(obsj1 % 2, d1, d2)
        # rho2a=Rotate(rho2,U)
        psi1a = np.dot(U, psi1)  #
        # if pp1>Product(rho2a,rho3):
        if pp1 > Product1(rho3, psi1a):
            U = U.conj().T
            #    rho2a=Rotate(rho2,U)
            psi1a = np.dot(U, psi1)  #
        while (Product1(rho3, psi1a) > pp1):
            # while (Product(rho2a, rho3) > pp1):
            # rho2b=rho2a
            psi1b = psi1a  #
            pp1 = Product1(rho3, psi1b)  #
            # pp1=Product(rho2b,rho3)
            psi1a = np.dot(U, psi1a)  #
            # rho2a=Rotate(rho2a,U)
    # return(rho2a)
    return (Project(psi1a))


# 3-partite entanglement
def Optimized3P(rho2, swaps, rho3, d1, i1):
    if i1 == 0:
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = RandomUBS(j1 % 2, d1, d1 * d1)
            #        rho2a=Rotate(rho2,U)
            psi1a = np.dot(U, psi1)  #
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                # rho2a=Rotate(rho2,U)
                psi1a = np.dot(U, psi1)  #
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 1:
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = Rotate(RandomUBS(j1 % 2, d1, d1 * d1), swaps[0])
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                # rho2a=Rotate(rho2,U)
                psi1a = np.dot(U, psi1)  #
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 2:
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = RandomUBS(j1 % 2, d1 * d1, d1)
            # rho2a=Rotate(rho2,U)
            psi1a = np.dot(U, psi1)  #
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                # rho2a=Rotate(rho2,U)
                psi1a = np.dot(U, psi1)  #
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    # return(rho2a)
    return (Project(psi1a))



# four partite entanglement
# swap12,swap13,swap23,swap34
def Optimized4P(rho2, rho3, swaps, d1, i1):
    if i1 == 0:
        # aBCD
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = RandomUBS(j1 % 2, d1, d1 ** 3)
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                # rho2a=Rotate(rho2,U)
                psi1a = np.dot(U, psi1)  #
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 1:
        # AbCD
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = Rotate(RandomUBS(j1 % 2, d1, d1 * d1 * d1), swaps[0])
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
            #    rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U,psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 2:
        # ABcD
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = Rotate(RandomUBS(j1 % 2, d1 ** 3, d1), swaps[3])
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
                # rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 3:
        # ABCd
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = RandomUBS(j1 % 2, d1 ** 3, d1)
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
                # rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 4:
        # abCD
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = RandomUBS(j1 % 2, d1 * d1, d1 * d1)
            # rho2a=Rotate(rho2,U)
            psi1a = np.dot(U, psi1)  #
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):  #
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
                # rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 5:
        # aBcD
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = Rotate(RandomUBS(j1 % 2, d1 * d1, d1 * d1), swaps[2])
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
                # rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 6:
        # aBCd
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate   (rho3[0]))  #
        for j1 in range(200):
            U = Rotate(RandomUBS(j1 % 2, d1 * d1, d1 * d1), swaps[1])
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
                # rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    # return(rho2a)
    return (Project(psi1a))

# UPB
def OptimizeUPB(rho2, rho3, rho4):
    pp1 = Product(rho2, rho3)
    whichline=0
    total=np.sum(np.abs(rho2[whichline]))
    while total==0:
        whichline+=1
        total = np.sum(np.abs(rho2[whichline]))
    psi1 = Normalize(np.conjugate(rho2[whichline]))#

    for obsj1 in range(400):
        U = RandomUUPB(rho4)
        # rho2a=Rotate(rho2,U)
        psi1a = np.dot(U, psi1)  #
        # if pp1>Product(rho2a,rho3):
        if pp1 > Product1(rho3, psi1a):
            U = U.conj().T
            #    rho2a=Rotate(rho2,U)
            psi1a = np.dot(U, psi1)  #
        while (Product1(rho3, psi1a) > pp1):
            # while (Product(rho2a, rho3) > pp1):
            # rho2b=rho2a
            psi1b = psi1a  #
            pp1 = Product1(rho3, psi1b)  #
            # pp1=Product(rho2b,rho3)
            psi1a = np.dot(U, psi1a)  #
            # rho2a=Rotate(rho2a,U)
    # return(rho2a)
    return (Project(psi1a))

# 4 qubits vs 3-sep
def Optimized4k3(rho2, rho3, swaps, d1, i1):
    if i1 == 0:
        # aBCD
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = RandomUBS(j1 % 2, d1, d1 ** 3)
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                # rho2a=Rotate(rho2,U)
                psi1a = np.dot(U, psi1)  #
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 1:
        # AbCD
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = Rotate(RandomUBS(j1 % 2, d1, d1 * d1 * d1), swaps[0])
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
            #    rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U,psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 2:
        # ABcD
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = Rotate(RandomUBS(j1 % 2, d1 ** 3, d1), swaps[1])
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
                # rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 3:
        # ABCd
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = RandomUBS(j1 % 2, d1 ** 3, d1)
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
                # rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)


# 5 qubits vs 4-separability
def Optimize54(rho2, rho3, swaps, d1, i1):
    if i1 == 0:
        # aBCDE
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = RandomUBS(j1 % 2, d1, d1 ** 4)
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                # rho2a=Rotate(rho2,U)
                psi1a = np.dot(U, psi1)  #
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 1:
        # AbCDE
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = Rotate(RandomUBS(j1 % 2, d1, d1 ** 4), swaps[0])
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
            #    rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U,psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 2:
        # ABcDE
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = Rotate(RandomUBS(j1 % 2, d1, d1 ** 4), swaps[1])
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
                # rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 3:
        # ABCDe
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = Rotate(RandomUBS(j1 % 2, d1 ** 4, d1), swaps[2])
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
            #    rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U,psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)

    if i1 == 4:
        # ABCDe
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = RandomUBS(j1 % 2, d1 ** 4, d1)
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
                # rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)

#continuar preguntando AQUI !!!11
# 6 qubits vs 5-separability
def Optimize6k5(rho2, rho3, swaps, d1, i1):
    if i1 == 0:
        # aBCDEF
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = RandomUBS(j1 % 2, d1, d1 ** 5)
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                # rho2a=Rotate(rho2,U)
                psi1a = np.dot(U, psi1)  #
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 1:
        # AbCDEF
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = Rotate(RandomUBS(j1 % 2, d1, d1 ** 5), swaps[0])
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
            #    rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U,psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 2:
        # ABcDEF
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = Rotate(RandomUBS(j1 % 2, d1, d1 ** 5), swaps[1])
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
            #    rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U,psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 3:
        # ABCdEF
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = Rotate(RandomUBS(j1 % 2, d1 ** 5, d1), swaps[2])
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
                # rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)
    if i1 == 4:
        # ABCDeF
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = Rotate(RandomUBS(j1 % 2, d1 ** 5, d1), swaps[3])
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
            #    rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U,psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)

    if i1 == 5:
        # ABCDEf
        pp1 = Product(rho2, rho3)
        psi1 = Normalize(np.conjugate(rho2[0]))  #
        for j1 in range(200):
            U = RandomUBS(j1 % 2, d1 ** 5, d1)
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
            # if pp1>Product(rho2a,rho3):
            if pp1 > Product1(rho3, psi1a):
                U = U.conj().T
                psi1a = np.dot(U, psi1)  #
                # rho2a=Rotate(rho2,U)
            # while(Product(rho2a, rho3)>pp1):
            while (Product1(rho3, psi1a) > pp1):
                psi1b = psi1a  #
                pp1 = Product1(rho3, psi1b)  #
                psi1a = np.dot(U, psi1a)  #
                # rho2b=rho2a
                # pp1=Product(rho2b,rho3)
                # rho2a=Rotate(rho2a,U)


# n qubits
def Optimize2FS(rho2, rho3, n):
    pp1 = Product(rho2, rho3)
    psi1 = Normalize(np.conjugate(rho2[0]))  #
    for j1 in range(200):
        U = RandomU2FS(n, j1 % n)
        psi1a = np.dot(U, psi1)
        # rho2a=Rotate(rho2,U)                                #
        # if pp1>Product(rho2a,rho3):
        if pp1 > Product1(rho3, psi1a):  #
            U = U.conj().T
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
        # while(Product(rho2a, rho3)>pp1):
        while (Product1(rho3, psi1a) > pp1):
            psi1b = psi1a  #
            pp1 = Product1(rho3, psi1b)  #
            psi1a = np.dot(U, psi1a)  #
            # rho2b=rho2a
            # pp1=Product(rho2b,rho3)
            # rho2a=Rotate(rho2a,U)
    # return(rho2a)
    return (Project(rho2a))


# n quDits
def OptimizedFS(rho2, rho3, ddd1, n):
    pp1 = Product(rho2, rho3)
    psi1 = Normalize(np.conjugate(rho2[0]))  #
    for j1 in range(200):
        U = RandomUdFS(ddd1, n, j1 % n)
        psi1a = np.dot(U, psi1)  #
        # rho2a=Rotate(rho2,U)
        # if pp1>Product(rho2a,rho3):
        if pp1 > Product1(rho3, psi1a):
            U = U.conj().T
            psi1a = np.dot(U, psi1)  #
            # rho2a=Rotate(rho2,U)
        # while(Product(rho2a, rho3)>pp1):
        while (Product1(rho3, psi1a) > pp1):
            psi1b = psi1a  #
            pp1 = Product1(rho3, psi1b)  #
            psi1a = np.dot(U, psi1a)  #
            # rho2b=rho2a
            # pp1=Product(rho2b,rho3)
            # rho2a=Rotate(rho2a,U)
    # return(rho2a)
    return (Project(psi1a))


# SWAP GATES at dimension d
def swap123(d):
    temp = np.zeros((d ** 3, d ** 3), dtype=complex)
    for i1 in range(d):
        for i2 in range(d):
            for i3 in range(d):
                temp[i1 * d ** 2 + i2 * d + i3][i2 * d ** 2 + i1 * d + i3] = 1
    return (temp)


# def swap133(d):
#     temp=np.zeros((d**3,d**3),dtype=complex)
#     for i1 in range(d):
#         for  i2 in range(d):
#             for i3 in range(d):
#                 temp[i1*d**2+i2*d+i3][i3*d**2+i2*d+i1]=1
#     return(temp)

# def swap233(d):
#     temp=np.zeros((d**3,d**3),dtype=complex)
#     for i1 in range(d):
#         for  i2 in range(d):
#             for i3 in range(d):
#                 temp[i1*d**2+i2*d+i3][i1*d**2+i3*d+i2]=1
#     return(temp)

def swap124(d):
    temp = np.zeros((d ** 4, d ** 4), dtype=complex)
    for i1 in range(d):
        for i2 in range(d):
            for i3 in range(d):
                for i4 in range(d):
                    temp[i1 * d ** 3 + i2 * d ** 2 + i3 * d + i4][i2 * d ** 3 + i1 * d ** 2 + i3 * d + i4] = 1
    return (temp)


def swap134(d):
    temp = np.zeros((d ** 4, d ** 4), dtype=complex)
    for i1 in range(d):
        for i2 in range(d):
            for i3 in range(d):
                for i4 in range(d):
                    temp[i1 * d ** 3 + i2 * d ** 2 + i3 * d + i4][i3 * d ** 3 + i2 * d ** 2 + i1 * d + i4] = 1
    return (temp)


# def swap144(d):
#     temp=np.zeros((d**4,d**4),dtype=complex)
#     for i1 in range(d):
#         for  i2 in range(d):
#             for i3 in range(d):
#                 for i4 in range(d):
#                     temp[i1*d**3+i2*d**2+i3*d+i4][i4*d**3+i2*d**2+i3*d+i1]=1
#     return(temp)

def swap234(d):
    temp = np.zeros((d ** 4, d ** 4), dtype=complex)
    for i1 in range(d):
        for i2 in range(d):
            for i3 in range(d):
                for i4 in range(d):
                    temp[i1 * d ** 3 + i2 * d ** 2 + i3 * d + i4][i1 * d ** 3 + i3 * d ** 2 + i2 * d + i4] = 1
    return (temp)


# def swap244(d):
#     temp=np.zeros((d**4,d**4),dtype=complex)
#     for i1 in range(d):
#         for  i2 in range(d):
#             for i3 in range(d):
#                 for i4 in range(d):
#                     temp[i1*d**3+i2*d**2+i3*d+i4][i1*d**3+i4*d**2+i3*d+i4]=1
#     return(temp)

def swap344(d):
    temp = np.zeros((d ** 4, d ** 4), dtype=complex)
    for i1 in range(d):
        for i2 in range(d):
            for i3 in range(d):
                for i4 in range(d):
                    temp[i1 * d ** 3 + i2 * d ** 2 + i3 * d + i4][i1 * d ** 3 + i2 * d ** 2 + i4 * d + i3] = 1
    return (temp)


def swap42():
    swap = [[1, 0, 0, 0],[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    return (Kronecker(swap, np.eye(4)))


def swap43():
    swap = [[1, 0, 0, 0],[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    return (Kronecker(np.eye(4)), swap)


def swap52():
    swap = [[1, 0, 0, 0],[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    return (Kronecker(swap, np.eye(8)))


def swap53():
    swap = [[1, 0, 0, 0],[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    swap1 = Kronecker(swap, np.eye(2))
    swap2 = Kronecker(np.eye(2), swap)
    swap3 = np.dot(np.dot(swap1, swap2), swap1)
    return (Kronecker(swap3, np.eye(4)))


def swap54():
    swap = [[1, 0, 0, 0],[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    return (Kronecker(np.eye(8), swap))


def swap62():
    swap = [[1, 0, 0, 0],[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    return (Kronecker(swap, np.eye(16)))


def swap63():
    swap = [[1, 0, 0, 0],[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    swap1 = Kronecker(swap, np.eye(2))
    swap2 = Kronecker(np.eye(2), swap)
    swap3 = np.dot(np.dot(swap1, swap2), swap1)
    return (Kronecker(swap3, np.eye(8)))


def swap64():
    swap = [[1, 0, 0, 0],[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    swap1 = Kronecker(swap, np.eye(2))
    swap2 = Kronecker(np.eye(2), swap)
    swap3 = np.dot(np.dot(swap1, swap2), swap1)
    return (Kronecker(np.eye(8), swap3))


def swap65():
    swap = [[1, 0, 0, 0],[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    return (Kronecker(np.eye(16), swap))


# Initialize files
# read rho
def Initrho0(prefix):
    #    return(scipy.io.mmread("_".join([prefix,"in.mtx"])))
    return (readmtx("_".join([prefix, "in.mtx"])))


# read or generate rho1
def Initrho1(prefix, rho, mode, d1, vis):
    if exists("".join([prefix, "_out_", str(vis), ".mtx"])):
        #        return(scipy.io.mmread("".join([prefix,"_out_",str(vis),".mtx"])))
        return (readmtx("".join([prefix, "_out_", str(vis), ".mtx"])))
    elif exists("".join([prefix, "_out_", str(mode), "_", str(d1), "_", str(vis), ".mtx"])):
        return (readmtx("".join([prefix, "_out_", str(mode), "_", str(d1), "_", str(vis), ".mtx"])))
    else:
        rhoa = np.zeros((len(rho), len(rho)), dtype=complex)
        for j1 in range(len(rho)):
            rhoa[j1][j1] = rho[j1][j1]
        return (rhoa)
    
#read unextendible basis
def readUPB(prefix):
    return(readmtx("".join([prefix,"_upb.mtx"])))
              
                   
#Swap states
def swapstates(rho1,rho2):
    rho5=rho2
    rho2=rho1
    rho1=rho5
    del(rho5)
    return [rho1,rho2]
                 


# Too slow convergence error    
def tooslow(counter, mode, prefix, vis,d1):
    with open("".join([prefix, "_abort_", str(mode), "_", str(d1), "_", str(vis), ".mtx"]), 'w') as file:
        file.write("The program has done 10% of planned trails, but it found only ")
        file.write(str(counter))
        file.write(" corrections and it was terminated.\n")
        file.write("Increase the number of trails or decrease the visibility.\n")
        file.close()


def showtooslow():
    if True:
        print(" ███▄    █ ▓█████ ██▒   █▓▓█████  ██▀███       ▄████  ██▓ ██▒   █▓▓█████     █    ██  ██▓███  ")
        if True:
            print(" ██ ▀█   █ ▓█   ▀▓██░   █▒▓█   ▀ ▓██ ▒ ██▒    ██▒ ▀█▒▓██▒▓██░   █▒▓█   ▀     ██  ▓██▒▓██░  ██▒")
            print("▓██  ▀█ ██▒▒███   ▓██  █▒░▒███   ▓██ ░▄█ ▒   ▒██░▄▄▄░▒██▒ ▓██  █▒░▒███      ▓██  ▒██░▓██░ ██▓▒")
        print("▓██▒  ▐▌██▒▒▓█  ▄  ▒██ █░░▒▓█  ▄ ▒██▀▀█▄     ░▓█  ██▓░██░  ▒██ █░░▒▓█  ▄    ▓▓█  ░██░▒██▄█▓▒ ▒")
        print("▒██░   ▓██░░▒████▒  ▒▀█░  ░▒████▒░██▓ ▒██▒   ░▒▓███▀▒░██░   ▒▀█░  ░▒████▒   ▒▒█████▓ ▒██▒ ░  ░")
        if True:
            print("░ ▒░   ▒ ▒ ░░ ▒░ ░  ░ ▐░  ░░ ▒░ ░░ ▒▓ ░▒▓░    ░▒   ▒ ░▓     ░ ▐░  ░░ ▒░ ░   ░▒▓▒ ▒ ▒ ▒▓▒░ ░  ░")
            print("░ ░░   ░ ▒░ ░ ░  ░  ░ ░░   ░ ░  ░  ░▒ ░ ▒░     ░   ░  ▒ ░   ░ ░░   ░ ░  ░   ░░▒░ ░ ░ ░▒ ░")
        print("   ░   ░ ░    ░       ░░     ░     ░░   ░    ░ ░   ░  ▒ ░     ░░     ░       ░░░ ░ ░ ░░")
        print("         ░    ░  ░     ░     ░  ░   ░              ░  ░        ░     ░  ░      ░")
    print("                       ░                                       ░")


# Wrong dimension    
def wrongdim(prefix, mode):
    with open("".join([prefix, "_wrong_dim_.txt"]), 'w') as file:
        file.write("Dimension of the input state not compatible with the declared mode")
        file.close()


# Read symmetry transformations
def DefineSym(prefix):
    symlist1 = []
    symlist2 = []
    symj1 = 0
    symj2 = 0
    if exists("_".join([prefix, "sym_0_0.mtx"])):
        symflag = True
    while exists("_".join([prefix, "sym", str(symj1), "0.mtx"])):
        while exists("".join([prefix, "_sym_", str(symj1), "_", str(symj2), ".mtx"])):
            #            symlist2.append(scipy.io.mmread("".join([prefix,"_sym_",str(symj1),"_",str(symj2),".mtx"])))
            symlist2.append(readmtx("".join([prefix, "_sym_", str(symj1), "_", str(symj2), ".mtx"])))
            symj2 = symj2 + 1
        symj2 = 0
        symlist1.append(symlist2)
        symj1 = symj1 + 1
    return (symlist1)


# Apply symmetries to a state
def ApplySym(rho, symlist1):
    asrho0 = rho
    for asj1 in range(len(symlist1)):
        for asj2 in range(len(symlist1[asj1])):
            asrho0 = asrho0 + Rotate(asrho0, symlist1[asj1][asj2])
        asrho0 = asrho0 / np.trace(asrho0)
    return (asrho0)


# Define projection
def DefineProj(projflag, prefix):
    if exists("_".join([prefix, "proj", ".mtx"])):
        projflag = True
        #        return(scipy.io.mmread("_".join([prefix,"proj.mtx"])))
        return (readmtx("_".join([prefix, "proj.mtx"])))
    else:
        return ([[]])


# Gilbert algorithm
def Gilbert(mode, prefix, vis, rho, steps, corrs, d1, d2, verboseflag):
    rho1 = Initrho1(prefix, rho, mode, d1, vis)
    if mode in (5,6):
        rho4=readUPB(prefix)
        NumOfStates=len(rho4)
    symflag = False
    symlist = DefineSym(prefix)
    if len(symlist) > 0:
        symflag = True
        print(len(symlist))
    projflag = False
    proj1 = DefineProj(projflag, prefix)
    # if symflag==True:
    #     rho1=ApplySym(rho1, symlist)
    if projflag == True:
        rho1 = Rotate(rho1, proj1)
    lastcorr = 0
    workinupb=0
    currentcorr = 0
    ll = []
    counter = 0
    flag = 0
    trail = 0
    if exists("".join([prefix, "_list_", str(vis), ".mtx"])) and exists("".join([prefix, "_out_", str(vis), ".mtx"])):
        #        ll2=scipy.io.mmread("".join([prefix,"_list_",str(vis),".mtx"]))
        ll2 = readmtx("".join([prefix, "_list_", str(vis), ".mtx"]))
        trail = int(ll2[len(ll2) - 1][0])
        counter = int(ll2[len(ll2) - 1][1])
        if mode in (5,6):
            workinupb=int(counter/500)%2
            if workinupb==1:
                rho,rho1=rho1,rho
        for i3 in range(len(ll2)):
            ll.append([int(ll2[i3][0]), int(ll2[i3][1]), ll2[i3][2]])
        if len(ll) == 1:
            currentcorr = ll[0][0]
        elif len(ll) > 1:
            currentcorr = ll[len(ll) - 1][0]
            lastcorr = ll[len(ll) - 2][0]
    if exists("".join([prefix, "_list_", str(mode), "_", str(d1), "_", str(vis), ".mtx"])) and exists(
            "".join([prefix, "_out_", str(mode), "_", str(d1), "_", str(vis), ".mtx"])):
        #        ll2=scipy.io.mmread("".join([prefix,"_list_",str(vis),".mtx"]))
        ll2 = readmtx("".join([prefix, "_list_", str(mode), "_", str(d1), "_", str(vis), ".mtx"]))
        trail = int(ll2[len(ll2) - 1][0])
        counter = int(ll2[len(ll2) - 1][1])
        for i3 in range(len(ll2)):
            ll.append([int(ll2[i3][0]), int(ll2[i3][1]), ll2[i3][2]])
        if len(ll) == 1:
            currentcorr = ll[0][0]
        elif len(ll) > 1:
            currentcorr = ll[len(ll) - 1][0]
            lastcorr = ll[len(ll) - 2][0]
    if steps < 0:
        steps = trail - steps
        steps = steps - steps % 10
    if corrs < 0:
        corrs = counter - corrs
    corrs = corrs - corrs % 50
    now = datetime.now()
    if verboseflag == True:
        print(now.strftime("%d/%m/%Y %H:%M:%S"), " Report: proceeding with", abs(steps) - trail, " iterations and ",
              abs(corrs) - counter, " corrections.")
    realflag = True
    for realcheck1 in range(len(rho)):
        for realcheck2 in range(len(rho)):
            if np.imag(rho[realcheck1][realcheck2]) != 0:
                realflag = False
                break
    if realflag == True and verboseflag == True:
        print("\n")
        print("Input state strictly real. Imaginary parts of the output state will be discarded")
    aa1 = Product(rho, rho)
    aa4 = 2 * Product(rho, rho1)
    aa6 = Product(rho1, rho1)
    rho3 = rho - rho1
    dd1 = Product(rho1, rho3)
    if mode in (3,5):
        swaps = [swap123(d1)]
    elif mode == 4:
        swaps = [swap124(d1), swap134(d1), swap234(d1), swap344(d1)]
    # else:
    #     swaps=[]
    if len(ll) == 0 or ll[-1][2] > 0.0000000000001:
        carryonflag = True;
    else:
        carryonflag = False
    while trail <= abs(steps) and counter <= abs(corrs) and (len(ll) == 0 or ll[-1][2] > 0.000000001):
        trail = trail + 1
#        print(trail," ",flag)
        if currentcorr - lastcorr > (steps - trail):
            if verboseflag == True:
                print("Too few steps left. Quitting")
            break
        if divmod(10 * trail / abs(steps), 1)[1] == 0:
            if counter < 50:
                tooslow(counter, mode, prefix, vis,d1)
                if verboseflag == True:
                    showtooslow()
                break
            now = datetime.now()
            if verboseflag == True:
                print(now.strftime("%d/%m/%Y %H:%M:%S"), " Report: done", trail, "/", abs(steps))
        # if mode==0:
        #     rho2=Random2FS(d1)
        if mode == 2:
            rho2 = RandomBS(d1, d2)
        elif mode == 1:
            rho2 = RandomdFS(d1, d2)
        elif mode == 3:
            rho2 = Random3P(d1, swaps, trail % 3)
        elif mode == 4:
            rho2 = Random4P(d1, swaps, trail % 7)
        elif mode == 5:
            if workinupb==0:
                rho2 = Random3P(3, swaps, trail % 3)
            else:
                rho2 = RandomUPB(rho4)
        elif mode == 5:
            if workinupb==0:
                rho2 = Random3P(3, swaps, trail % 3)
            else:
                rho2 = RandomUPB(rho4)
        elif mode == 5:
            if workinupb==0:
                rho2 = Random3P(3, swaps, trail % 3)
            else:
                rho2 = RandomUPB(rho4)
        else:
            if verboseflag == True:
                print("Mode ", mode, "does not exist!")
            DisplayHelp()
            break
        if Product(rho2, rho3) > dd1:
            # if mode==0:
            #     rho2=Optimize2FS(rho2, rho3, d1)
            if mode == 2:
                rho2 = OptimizeBS(rho2, rho3, d1, d2)
            elif mode == 1:
                rho2 = OptimizedFS(rho2, rho3, d1, d2)
            elif mode == 3:
                rho2 = Optimized3P(rho2, swaps, rho3, d1, trail % 3)
            elif mode == 4:
                rho2 = Optimized4P(rho2, rho3, swaps,d1, trail % 7)
            elif mode == 5:
                if workinupb == 0:
                    rho2 = Optimized3P(rho2, swaps, rho3, d1, trail % 3)
                else:
                    rho2 = OptimizeUPB(rho2, rho3, rho4)
            elif mode == 6:
                if workinupb == 0:
                    rho2 = OptimizeFS(rho2, rho3, d1, d2)
                else:
                    rho2 = OptimizeUPB(rho2, rho3, rho4)
            if realflag == True:
                rho2 = np.real(rho2)
            if symflag == True:
                rho1 = ApplySym(rho1, symlist)
            if projflag == True:
                rho1 = Rotate(rho1, proj1)
                rho1 = rho1 / np.trace(rho1)
            aa3 = Product(rho2, rho2)
            aa2 = 2 * Product(rho, rho2)
            aa5 = 2 * Product(rho1, rho2)
            #            bb1=aa1-aa2+aa3
            bb2 = -aa4 + aa2 + aa5 - 2 * aa3
            bb3 = aa6 - aa5 + aa3
            cc1 = -bb2 / (2 * bb3)
            #            cc2=-bb2*bb2/(4*bb3)+bb1
            if 0 <= cc1 and cc1 <= 1:
                rho1 = cc1 * rho1 + (1 - cc1) * rho2
                if symflag == True and counter % 50 == 0:
                    rho1 = ApplySym(rho1, symlist)
                counter = counter + 1
                if counter % 50 in (10, 20, 30, 40) and verboseflag == True:
                    0
                    # marker=counter%50
                    #print(Product(rho3, rho3))
                    # if marker==0:
                    #   print("")
                    # if marker in [2,4,6,8]:
                    #     # print("\b",end="")
                    #     # print("\b",end="")
                    #     # print("",end="")
                    #     print(marker,end=" ")
                    # elif marker==0:
                    #     print("")
                    # #     print("\b",end="")
                    # #     print("\b",end="")
                    # else:
                    #     # print("\b",end="")
                    #     # print("\b",end="")
                    #     print(marker,end=" ")
                rho3 = rho - rho1
                aa4 = 2 * Product(rho, rho1)
                aa6 = Product(rho1, rho1)
                dd1 = aa4 / 2 - aa6
                flag = 1
            if 0 > cc1 or cc1 > 1:
                flag = 0
#            print(cc1)
            if counter % 50 == 0 and flag == 1:
                current_distance=Product(rho3,rho3)
                if verboseflag:
                    now = datetime.now()
                    #AQUI PER VEURE LES COSES
                    #print(now.strftime("%H:%M:%S: "), "Trails:", trail, " Corrections:", counter, "D^2:",current_distance)
                if counter % 500 == 0:
                    makeshortreport(prefix, ll, mode, d1, vis, counter,current_distance)
                ll.append([trail, counter, current_distance])
                if workinupb==0:
                #                scipy.io.mmwrite("".join([prefix,"_out_",str(vis),".mtx"]),rho1)
                    writemtx("".join([prefix, "_out_", str(mode), "_", str(d1), "_", str(vis), ".mtx"]), rho1, 2)
                #                scipy.io.mmwrite("".join([prefix,"_list_",str(vis),".mtx"]),ll)
                    writemtx("".join([prefix, "_list_", str(mode), "_", str(d1), "_", str(vis), ".mtx"]), ll, 1)
           ###############################
                else:
                    writemtx("_".join([prefix,"in.mtx"]),rho1,2)
                if (counter % 500)==0 and mode == 5:
                    if verboseflag:
                        print("Swapping states")
                    rho1,rho=rho,rho1
                    aa1 = Product(rho, rho)
                    aa1 = Product(rho, rho)
                    aa4 = 2 * Product(rho, rho1)
                    aa6 = Product(rho1, rho1)
                    rho3 = rho - rho1
                    dd1 = Product(rho1, rho3)
                    workinupb=1-workinupb
                lastcorr = currentcorr
                currentcorr = trail
                flag=0
    return (ll)


def invert(c, a):
    return (1 / (c - a))


def mean(l):
    return (sum(l) / len(l))


def R(l, a):
    ll1 = list(map(lambda x1: invert(x1, a), l))
    return (mean(list(map(lambda x1, x2: x1 * x2, ll1, list(range(len(l)))))) - mean(ll1) * mean(
        list(range(len(l))))) / math.sqrt((mean(list(map(lambda x: x ** 2, ll1))) - mean(ll1) ** 2) * (
                mean(list(map(lambda x: x ** 2, list(range(len(l)))))) - mean(list(range(len(l)))) ** 2))


def listshift(l1, a1):
    return (list(map(lambda x: x - a1, l1)))


def cov(l1, l2):
    return (mean(list(map(lambda x1, x2: x1 * x2, listshift(l1, mean(l1)), listshift(l2, mean(l2))))))


def trend(l1, l2):
    l1a = list(map(lambda x: math.log(x), l1))
    l2a = list(map(lambda x: math.log(x), l2))
    return (cov(l1a, l2a) / cov(l1a, l1a))


def offset(l1, l2):
    l1a = list(map(lambda x: math.log(x), l1))
    l2a = list(map(lambda x: math.log(x), l2))
    return (mean(l2a) - mean(l1a) * trend(l1, l2))


def findmaximum(ll):
    list1 = list(map(lambda x: x[2], ll))
    list2 = []
    for j1 in range(int(len(list1) / 2), len(list1)):
        list2.append(list1[j1])
    aaa1 = list2[len(list2) - 1] - .000001
    step1 = aaa1 / 10000
    while R(list2, aaa1 - step1) > R(list2, aaa1) and aaa1 > 0:
        aaa1 = aaa1 - step1
    return (aaa1)


def makeshortreport(prefix, ll, mode, d1, vis,trail,current_distance):
    ll10 = list(map(lambda x: x[0], ll))
    ll11 = list(map(lambda x: x[1], ll))
    kk = findmaximum(ll)
    ll12 = []
    for j1 in range(int(2 * len(ll) / 3), len(ll)):
        ll12.append(ll[j1][2])
    #print("Basing on decay, the squared HS distance is estimated to be ", str(kk))
    with open("".join([prefix, "_report_", str(mode), "_", str(d1), "_", str(vis),"_", ".txt"]), 'a') as file:
        file.write("Basing on decay, the squared HS distance is estimated to be ")
        file.write(str(kk))
        file.write(" (R=")
        file.write(str(R(ll12, kk)))
        file.write(")\n")
        file.write("The dependence between corrs and trail is approximately:\n")
        file.write("corr=trail^")
        file.write(str(trend(ll10, ll11)))
        file.write("*")
        file.write(str(math.exp(offset(ll10, ll11))))
        file.write("\n-----------------\n")
        file.write("The current squared distance is {}.".format(current_distance))
        file.write("\n-----------------\n")
        if kk>0.6*current_distance:
                file.write("Witness possible. \n")
        file.write("Report prepared for {} corrections".format(trail))
        file.write("\n-----------------\n")
        file.close()
    return R(ll12, kk),kk


def makelongreport(prefix, mode, vis, swaps, d1, d2, ll, verboseflag):
    # optw=OptimizeW(prefix, mode, vis, swaps, d1, d2,verboseflag)
    # wdist0=WitnessDist(prefix, vis, optw,verboseflag)
    # with open("".join([prefix,"_report_",str(vis),".txt"]),'a') as file:
    #     if wdist0==-2:
    #         file.write("The algorithm did not yield a valid entanglement witness.")
    #     else:
    #         file.write("The squared distance based on entanglement witness is ")
    #         file.write(str(wdist0))
    #     file.close()
    rhoa = readmtx("".join([prefix, "_in.mtx"]))
    #    rhob=scipy.io.mmread("".join([prefix,"_out_",str(vis),".mtx"]))
    rhob = readmtx("".join([prefix, "_out_", str(mode), "_", str(d1), "_", str(vis), ".mtx"]))
    rhoa = vis * rhoa + (1 - vis) * np.eye(len(rhoa)) / len(rhoa)
    witness = rhoa - rhob
    writemtx("".join([prefix, "_witness_", str(mode), "_", str(d1), "_", str(vis), ".mtx"]), witness, 2)


def OptimizeW(prefix, mode, vis, swaps, d1, d2, verboseflag):
    l = -1
    #    rhoa=scipy.io.mmread("_".join([prefix,"in.mtx"]))
    rhoa = readmtx("_".join([prefix, "in.mtx"]))
    rhob = Initrho1(prefix, rhoa, vis)
    witness = rhoa - rhob
    #     zasieg=1000*d1
    if mode == 2:
        zasieg = 1500
    if mode == 1:
        zasieg = 1500
    if mode == 3:
        zasieg = 1500
    if mode == 4:
        zasieg = 1500
    if verboseflag == True:
        print("Optimizing the potential witness operator. Number of trails:", zasieg)
    for owi1 in range(zasieg):
        # if mode==0:
        #     w1=Random2FS(d1)
        #     l.append(Optimize2FS(w1,witness, d1))
        if owi1 % 100 == 0 and verboseflag:
            print("trail:", owi1)
        if mode == 2:
            w1 = RandomBS(d1, d2)
            rho2a = OptimizeBS(w1, witness, d1, d2)
            l1 = Product(rho2a, witness)
            if l1 > l:
                l = l1
        if mode == 1:
            w1 = RandomdFS(d1, d2)
            rho2a = OptimizedFS(w1, witness, d1, d2)
            l1 = Product(rho2a, witness)
            if l1 > l:
                l = l1
        if mode == 3:
            for owi2 in range(3):
                w1 = Random3P(d1, swaps[0], owi2)
                rho2a = Optimize2FS(w1, witness, d1)
                l1 = Product(rho2a, witness)
                if l1 > l:
                    l = l1
        if mode == 4:
            for owi2 in range(7):
                w1 = Random4P(d1, swaps[0], swaps[1], swaps[2], swaps[3], owi2)
                rho2a = Optimize2FS(w1, witness, d1)
                l1 = Product(rho2a, witness)
                if l1 > l:
                    l = l1
    return (l)


def WitnessDist(prefix, vis, sepmax, verboseflag):
    #    rhoa=scipy.io.mmread("".join([prefix,"_in.mtx"]))
    rhoa = readmtx("".join([prefix, "_in.mtx"]))
    #    rhob=scipy.io.mmread("".join([prefix,"_out_",str(vis),".mtx"]))
    rhob = readmtx("".join([prefix, "_out_", str(mode), "_", str(d1), "_", str(vis), ".mtx"]))
    rhoa = vis * rhoa + (1 - vis) * np.eye(len(rhoa)) / len(rhoa)
    witness = rhoa - rhob
    wdist = (Product(witness, rhoa) - sepmax) / math.sqrt(Product(witness, witness))
    if wdist < 0:
        if verboseflag == True:
            print("No entanglement witness found.")
    else:
        if verboseflag == True:
            print("Witness-based estimated squared distance:", wdist ** 2, " (VERIFY!!!)")
        #        scipy.io.mmwrite("".join([prefix,"_witness_",str(vis),".mtx"]),witness,"".join(["Estimated sqared distance:",str(wdist**2)]))
        writemtx("".join([prefix, "_witness_", str(mode), "_", str(d1), "_", str(vis), ".mtx"]), witness, 2)
    return (wdist ** 2)


def DetectDim0(mode, totaldim, verboseflag):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
              107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
              227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347,
              349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
              467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607,
              613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743,
              751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883,
              887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031,
              1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151,
              1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279,
              1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423,
              1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523,
              1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627,
              1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777,
              1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907,
              1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999]
    ddflag = False
    if mode == 1:
        for pr in primes:
            if math.log(totaldim, pr) == int(math.log(totaldim, pr)):
                if verboseflag == True:
                    print("Determined size: ", pr, " number of subsystems:", int(math.log(totaldim, pr)))
                return ([pr, int(math.log(totaldim, pr))])
                ddflag = True
                break
    elif mode == 2:
        if math.sqrt(totaldim) == int(math.sqrt(totaldim)):
            pr = int(math.sqrt(totaldim))
            if verboseflag:
                print("Determined sizes of subsytems: ", pr, int(totaldim / pr))
            return ([int(math.sqrt(totaldim)), int(math.sqrt(totaldim))])
            ddflag = True
        else:
            for pr in primes:
                if totaldim % pr == 0:
                    if verboseflag:
                        print("Determined sizes of subsytems: ", pr, int(totaldim / pr))
                    return ([pr, int(totaldim / pr)])
                    ddflag = True
                    break
    elif mode == 3:
        if totaldim ** (1. / 3) == int(totaldim ** (1. / 3)):
            if verboseflag:
                print("Determined size: ", int(totaldim ** (1. / 3)), " number of subsystems:", 3)
            return ([int(totaldim ** (1. / 3)), 3])
            ddflag = True
    elif mode == 4:
        if totaldim ** (1. / 4) == int(totaldim ** (1. / 4)):
            if verboseflag:
                print("Determined size: ", int(totaldim ** (1. / 4)), " number of subsystems:", 4)
            return ([int(totaldim ** (1. / 4)), 4])
            ddflag = True
    elif mode in (5,6):
        if totaldim==27:
            if verboseflag:
                print("Determined size: 3, number of subsystems: 3")
            return [3,3]
            ddflag=True


def DetectDim1(mode, totdim, d1, verboseflag):
    ddflag = False
    if mode == 1:
        if math.log(totdim, d1) == int(math.log(totdim, d1)):
            ddflag = True
            if verboseflag == True:
                print("Determined size: ", d1, " number of subsystems:", int(math.log(totdim, d1)))
            return (int(math.log(totdim, d1)))
    elif mode == 2:
        if totdim / d1 == int(totdim / d1):
            ddflag = True
            if verboseflag == True:
                print("Determined sizes", d1, ", ", int(totdim / d1))
            return (int(totdim / d1))
    elif mode == 3:
        if d1 ** 3 == totdim:
            ddflag = True
            if verboseflag:
                print("Determined size: {}, number of subsystems: 3".format(d1))
            return (3)
    elif mode == 4:
        if d1 ** 4 == totdim:
            ddflag = True
            if verboseflag:
                print("Determined size: {}, number of subsystems: 4".format(d1))
            return (4)
    elif mode in (5,6):
        if 27 == totaldim:
            ddflag = True
            if verboseflag:
                print("Determined size: {}, number of subsystems: 3".format(d1))
            return (3)
    if ddflag == False:
        return (0)



def frobeniusnorm(A):
    if type(A)=='Tensor':
        A=A.np()
    return(np.sqrt(np.trace(A@A.conj().T)))

def W_opt(rho,rhos):
    coefff1=frobeniusnorm(rho-rhos)
    coefff2=np.trace(rhos@(rhos-rho))
    return (rhos-rho-coefff2*id(len(rho)))/coefff1


def entlabelsifter(entanglementlabel,recovered_states):
    sep=[]
    ent=[]
    npi=[]
    for i in range(len(entanglementlabel)):
        if entanglementlabel[i]==0.:
            sep.append(recovered_states[i])
        elif entanglementlabel[i]==1.:
            ent.append(recovered_states[i])
        elif entanglementlabel[i]==-100.:
            npi.append(recovered_states[i])
    return(ent,sep,npi)



def closestsep(rho, listrhos):
    aold=1000
    for i in range(len(listrhos)):
        if type(listrhos[i])!=type(rho):
            listrhos[i]=torch.from_numpy(listrhos[i])
        anew=frobeniusnorm(listrhos[i]-rho)
        if anew<aold:
            aold=anew
            estatsep=listrhos[i]
    return estatsep





def optimalwitnesslist(dim1,dim2):
    optwitlist=[]
    for i in range(len(rhoent)):
        rhosep=closestsep(rhoent[i],rhoseplist)
        optwitlist.append(rhoent[i],rhosep)
    return optwitlist








def DisplayHelp():
    print("CSSFinder mode verbose prefix vis steps corrs [d1]")
    print("mode=1: full separability of an n-quDit state")
    print("mode=2: separability of a bipirtite state")
    print("mode=3: genuine 3-partite entaglement of a 3-quDit state")
    print("mode=4: genuine 4-partite entaglement of a 4-quDit state")
    print("mode=5: genuine 3-partite entanglement of a 3-qbit subspace (defined by rows in prefix_upb.mtx)")
    print("verbose=0: supress on-screen ouptut")
    print("verbose=1: show on-screen messages and reports")
    print("prefix: prefix of all used files")
    print("vis: visibility against white noise. Between 0 and 1. To be used when the algorithm is stuck")
    print(
        "steps: number of attempts to correct the closest separable state. A negative number adds its value to pre-existing list of results")
    print("steps are rounded down to a multiple of 10")
    print(
        "corrs: the maximal number of corrections to the closest separable state. A negative number adds its value to pre-existing list of results.")
    print("corrs are rounded down to a multiple of 50")
    print(
        "d1: the dimension of the first subsystem. Dimension of the other subsystem or the number of parties is deduced from the dimension of the input state.")
    print("Input:")
    print("prefix_in.mtx: the input state in MTX format")
    print(
        "prefix_sym_0_0.mtx,prefix_sym_0_1.mtx,...: symmetry unitaries applied to the output state in MTX format. The first number is the symmetry label, the second is the manifold. Optional")
    print("prefix_proj.mtx,prefix_sym_0_1.mtx,...: projections applied to the output state in MTX format. Optional")
    print("Output:")
    print("prefix_out_mode_d1_vis.mtx: final separable state (can be used as an initial separable state)")
    print("prefix_list_mode_d1_vis.mtx: number of steps, corrections, and the squared HS distance every 50 corrections")
    print("If these files exist, the program will resume from the last record")
    print("prefix_report_mode_d1_vis.txt: The report file.")
    #    print("prefix_witness_mode_d1_vis.txt: entanglement witness candidate. Consult [Quantum Reports 2, 49].")
    print(
        "prefix_abort.txt: The error message if the algorithm was extremally slow (for some highly entangled states).")


# Display the Logo
def DisplayLogo():
    print(" ██████╗███████╗███████╗███████╗██╗███╗   ██╗██████╗ ███████╗██████╗")
    if True:
        print("██╔════╝██╔════╝██╔════╝██╔════╝██║████╗  ██║██╔══██╗██╔════╝██╔══██╗")
        if True:
            print("██║     ███████╗███████╗█████╗  ██║██╔██╗ ██║██║  ██║█████╗  ██████╔╝")
        print("██║     ╚════██║╚════██║██╔══╝  ██║██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗")
    print("╚██████╗███████║███████║██║     ██║██║ ╚████║██████╔╝███████╗██║  ██║")
    print(" ╚═════╝╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝")


def main(args):
    sii=False
    #     argnum=len(sys.argv)
    #     args=argstring
    argnum = len(args)
    verboseflag = True
    #     args=sys.argv
    argflag = False
    symflag = False
    if (argnum == 7 or argnum == 8) and (int(args[1]) in range(1, 6)):
        argflag = True
    if argflag == False:
        DisplayHelp()
    else:
        report = int(args[2])
        if report == 0:
            verboseflag = False
        if verboseflag == True:
            DisplayLogo()
        correctdimflag = True
        mode = int(args[1])
        prefix = args[3]
        vis = float(args[4])
        rho = Initrho0(prefix)
        totdim = len(rho)
        if argnum == 7:
            [d1, d2] = DetectDim0(mode, totdim, verboseflag)
        elif argnum == 8:
            d1 = int(args[7])
            d2 = DetectDim1(mode, totdim, d1, verboseflag)
        if d2 == 0:
            correctdimflag == False
            wrongdim(prefix, mode)
        else:
            rho = vis * rho + (1 - vis) * np.eye(totdim) / totdim
            rho1 = np.zeros((totdim, totdim))
            steps = int(args[5])
            if steps == 0:
                steps = -30000000
            corrs = int(args[6])
            if corrs == 0:
                if exists("".join([prefix, "_list_", str(mode), "_", str(d1), "_", str(vis), ".mtx"])) and exists(
                        "".join([prefix, "_out_", str(mode), "_", str(d1), "_", str(vis), ".mtx"])):
                    corrs = -500
                else:
                    coors = 2500
        if correctdimflag == True:
            ll1 = Gilbert(mode, prefix, vis, rho, steps, corrs, d1, d2, verboseflag)
            if len(ll1) < 10:
                if verboseflag == True:
                    print("The report can be generated only if more than 500 corrections were performed.")
            else:
                sii=True
                #print(ll1[-1][2],ll1)
                rr=makeshortreport(prefix, ll1, mode, d1, vis, steps,ll1[-1][2])
                if mode == 3:
                    swaps = [swap123(d1)]
                elif mode == 4:
                    swaps = [swap124(d1), swap134(d1), swap234(d1), swap344(d1)]
                elif mode == 5:
                    swaps = [swap123(3)]
                else:
                    swaps = []
                makeshortreport(prefix, ll1, mode, d1, vis, steps,ll1[-1][2])
        #                    makelongreport(prefix, mode, vis, swaps, d1, d2, ll1,verboseflag)
        else:
            print("Input state dimensionality incompatiblie with declared mode.")
            DisplayHelp()
    
    if sii==True:
        print(rr[1],rr[0])
        return rr[1],rr[0]
    else:
        return 0,0

#if __name__ == '__main__':
    #main(sys.argv)








def entanglement_label(rho,d1,d2): #RETURNS STRING TRUE IF ENTANGLED, RETURNS STRING FALSE IF SEPARABLE AND RETURNS STRING 0 IF NO IDEA. SAVES THE CORRESPONDING MATRICES IN A LIST FOR ENTANGLED, SEPARABLE AND NPI
    layers=1
    rhopt=partial_transpose(rho,d1,d2)
    if is_positive_semidefinite(rhopt)==False:
        #print('rho is NPT entangled')
        return 1.0
    elif is_positive_semidefinite(rhopt)==True:
        #print('rho is PPT')
        c=CCNR_criterion(rho, d1, d1)
        if c==True:
            #print('CNNR says rho is entangled')
            return 1.0
        elif c==False:
            #print('CCNR is not conclusive')
            d=Symm_Extension(rho, d1, d1, layers)
            if d==True:
                #print('Symm Extension says rho is entangled')
                return 1.0
            elif d==False:
                #print('Symm Extension is not conclusive')
                
                writemtx('matrices_in.mtx',rho.numpy(),2)
                sep=main(['hi',2,1,'matrices',1,10**5,10**4,d1])
                os.remove('matrices_in.mtx')
                os.remove('matrices_list_2_3_1.0.mtx')
                
                if sep[0]<=0 and sep[1]>=0.97:
                    #print('CSFINDER says rho is separable')
                    return 0.
                else:
                    print('PPT, CCNR, DPS extension and CSSFINDER are inconclusive')
                    return -100.0
                    
            
  




'''
from criterion_utils import (
    is_entangled_ppt,
    mutual_information_sign_ent,
    mutual_information_values_ent,
)
'''


#SIMULATION CODE BEGINS HERE
'''
a=generate_states(2,2,1)
print(a)
'''

def generate_states(dim1, dim2, n_rhos=1000):
    """Generates quantum states using random rho.
    Returns a list of generated states."""

    return [rand_rho(dim1, dim2) for _ in tqdm(range(n_rhos), desc="Generating states")]




def generate_entanglement_labels(sample_states, dim1, dim2):
    """Generates entanglement labels for the states.
    Returns a tensor of entanglement labels."""

    n_rhos = len(sample_states)
    is_entangled = torch.zeros((n_rhos,), dtype=torch.float64).to(device)
    for i in tqdm(range(n_rhos), desc="Generating entanglement labels"):
        rho = sample_states[i]
        
        is_entangled[i] = torch.tensor(entanglement_label(rho, dim1, dim2), dtype=torch.float64)#.clone().detach().to(device)  # type: ignore
    return is_entangled


def compute_trace_values(n_witnesses, states, dim1, dim2, get_witness_from_family):
    """Computes trace values for each witness.
    Returns a list of trace values for each witness."""

    trace_values_list = []
    for _ in tqdm(range(n_witnesses)):
        witness = get_witness_from_family(dim1, dim2).view(1, -1).to(device).conj()
        value = (witness @ states).squeeze().real
        trace_values_list.append(value)
    return trace_values_list


def compute_information(
    n_witnesses, states, is_entangled, dim1, dim2, get_witness_from_family
):
    """Computes mutual information for entanglement.
    Returns lists of fine and coarse grained information."""

    info_fine_grained = []
    info_coarse_grained = []
    
    for _ in tqdm(range(n_witnesses)):
        witness = get_witness_from_family(dim1, dim2).view(1, -1).to(device).conj()
        trace_values = (witness @ states).squeeze().real
        info_fine_grained.append(mutual_information_values_ent(trace_values, is_entangled).item())
        info_coarse_grained.append(mutual_information_sign_ent(trace_values, is_entangled).item())
        
    return info_fine_grained, info_coarse_grained


def plot_histograms(info_fine_grained, info_coarse_grained, filename):
    """Plots histograms of fine and coarse grained information.
    Not used in the paper, but can be used to visualize the data."""

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"{filename}")
    ax[0].set_title("Trace Value information")
    ax[0].xaxis.set_label_text("Mutual Information")
    ax[0].set_xlim([0, 0.1])

    ax[1].set_title("Trace Sign information")
    ax[1].xaxis.set_label_text("Mutual Information")
    ax[1].set_xlim([0, 0.012])

    ax[0].hist(info_fine_grained, bins=100)
    ax[1].hist(info_coarse_grained, bins=100)
    if not os.path.exists("histograms"):
        os.makedirs("histograms")
    plt.savefig(f"histograms/{filename}")
    plt.close(fig)

def simulate_fixed_parameters(
    n_witnesses,
    states,
    is_entangled,
    dim1,
    dim2,
    get_witness_from_family,
    filename="histograms.png",
):
    """Performs the simulation by generating states,
    computing mutual information, and plotting histograms."""

    info_fine_grained, info_coarse_grained = compute_information(
        n_witnesses, states, is_entangled, dim1, dim2, get_witness_from_family
    )
    # Optional, not needed for the paper:
    # plot_histograms(info_fine_grained, info_coarse_grained, filename)

    # Save each individual mutual information value into separate CSV files
    if not os.path.exists("data"):
        os.makedirs("data")

    with open(
        f"data/{filename.replace('.png', '_fine_grained.csv')}", "w", newline=""
    ) as file:
        writer = csv.writer(file)
        # writer.writerow(['Info_Fine_Grained'])
        for fine in info_fine_grained:
            writer.writerow([fine])

    with open(
        f"data/{filename.replace('.png', '_coarse_grained.csv')}", "w", newline=""
    ) as file:
        writer = csv.writer(file)
        # writer.writerow(['Info_Coarse_Grained'])
        for coarse in info_coarse_grained:
            writer.writerow([coarse])


def save_example_joint_events_distribution_data(is_entangled, states):
    """Specific for the qubit-qubit case"""
    witness = torch.tensor([[
        1/4, 0, 0, 1/2,
        0, 1/4, 0, 0,
        0, 0, 1/4, 0,
        1/2, 0, 0, 1/4,
        ]], dtype=torch.complex128).view(1, -1).to(device).conj()
    
    trace_values = (witness @ states).squeeze().real
    
    with open("data/example_joint_events.csv", "w") as file:
        writer = csv.writer(file)
        for i in range(len(trace_values)):
            writer.writerow([trace_values[i].item(), is_entangled[i].item()])
        
def save_entanglement_events(is_entangled, n1, n2):
    """Saves the entangled distribution for a given dimension"""
    with open(f"data/entanglement_events_{n1}_{n2}.csv", "w") as file:
        writer = csv.writer(file)
        for i in range(len(is_entangled)):
            writer.writerow([is_entangled[i].item()])
    
    
    
    
    
    

def optimalwitness(rhoent,rhoseplist):
    optwitlist=[]
    for i in range(1000): #INPUT1
        veca=np.random.uniform(-1, 1, (3,)) + 1.j * np.random.uniform(-1, 1, (3,))
        vecb=np.random.uniform(-1, 1, (3,)) + 1.j * np.random.uniform(-1, 1, (3,))
        veca=veca/(np.sqrt(veca@np.conjugate(veca)))
        vecb=vecb/(np.sqrt(vecb@np.conjugate(vecb)))
        rhosepextra=np.kron(np.outer(veca,veca.conj().T),np.outer(vecb,vecb.conj().T))
        rhoseplist.append(rhosepextra)
    for i in range(len(rhoent)):
        rhosep=closestsep(rhoent[i],rhoseplist)
        optwitlist.append(W_opt(rhoent[i],rhosep))
    return optwitlist[np.random.randint(0,len(optwitlist)-1)]



def llistasep(N):
    rhoseplist=[]
    for i in range(N):
        veca=np.random.uniform(-1, 1, (3,)) + 1.j * np.random.uniform(-1, 1, (3,))
        vecb=np.random.uniform(-1, 1, (3,)) + 1.j * np.random.uniform(-1, 1, (3,))
        veca=veca/(np.sqrt(veca@np.conjugate(veca)))
        vecb=vecb/(np.sqrt(vecb@np.conjugate(vecb)))
        rhosepextra=np.kron(np.outer(veca,veca.conj().T),np.outer(vecb,vecb.conj().T))
        rhoseplist.append(rhosepextra)
    return rhoseplist



def optimalwitness1(rhoent,rhoseplist):
    optwitlist=[]
    for i in range(len(rhoent)):
        rhosep=closestsep(rhoent[i],rhoseplist)
        optwitlist.append(W_opt(rhoent[i],rhosep))
    return optwitlist






def simulate_all_parameters():
    """Performs the simulation for all combinations of
    dimensions, witness generators, and powers."""

    n_witnesses = 10000 #INPUT2
    n_states = 10000 #INPUT3
    dimensions_system1 = [3]
    dimensions_system2 = [3]
    functional_generators = [
        random_functional,
        #random_witness_from_partial_transpose,
        #random_optimal3x3witness,
        optimalwitness
        #random_witness_from_family,
    ]
    powers = [1]
    
    # Keep track of total number of simulations
    total_simulations = len(dimensions_system1) * len(dimensions_system2) * len(functional_generators) * len(powers)
    simulation_count = 1
    start_time = time.time()

    # Iterate over all combinations of parameters:
    
    # System dimensions
    for n1 in dimensions_system1:
        for n2 in dimensions_system2:
            if n1 == n2 == 4:
                continue
            
            # Generate states and entanglement labels
            print(f"\n\nRunning simulation for {n1}x{n2} systems.")
            states = (
                torch.stack(generate_states(n1, n2, n_states))
                .to(device)
                .view(n_states, -1, 1)
            )
            states1 = states.squeeze(-1)  # Now shape: (n_states, n1 * n2)
            recovered_states = states1.view(n_states, n1*n2, n2*n1)
            is_entangled = generate_entanglement_labels(recovered_states, n1, n2)
            print(is_entangled)
            print(len(recovered_states),len(is_entangled))
            print(min(is_entangled))
            rhoent,rhoseplist,rhonpilist=entlabelsifter(is_entangled,recovered_states)
            print(rhoseplist)
            # Save sample joint probability distribution data for n1 = n2 = 2
            if (n1 == n2 == 2):
                save_example_joint_events_distribution_data(is_entangled, recovered_states)
            # Save entangled distribution data
            save_entanglement_events(is_entangled, n1, n2)

            # Observable Families
            for functional_generator in functional_generators:
                # Possibilities for higher moments
                for power in powers:
                    if (
                        functional_generator.__name__ == "random_witness_from_family"
                        and not (n1 == n2 == 3)
                    ):
                        continue

                    # Print partial elapsed time
                    elapsed_time = time.time() - start_time
                    hours, remainder = divmod(elapsed_time, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print(
                        f"\nRunning simulation {simulation_count} out of {total_simulations}. Elapsed time: {int(hours):02}:{int(minutes):02}:{seconds:02.0f}."
                    )

                    # Run simulation for given moment
                    def observable_generator_given_power(dim1, dim2):
                        if functional_generator==optimalwitness:
                            rhoseplist=llistasep(10)
                            w = functional_generator(rhoent, rhoseplist)
                            out = w.clone()
                            for _ in range(power - 1):
                                print(out-w)
                                out = out @ w
                            return out
                            
                        else:
                            w = functional_generator(dim1, dim2)
                            out = w.clone()
                            for _ in range(power - 1):
                                print(out-w)
                                out = out @ w
                            return out
                    
                    simulate_fixed_parameters(
                        n_witnesses,
                        states,
                        is_entangled,
                        n1,
                        n2,
                        observable_generator_given_power,
                        f"histograms_{n1}_{n2}_{functional_generator.__name__}_{power}th_momentum.png",
                    )
                    simulation_count += 1
                    
    # Print total elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nSimulation finished. Elapsed time: {int(hours):02}:{int(minutes):02}:{seconds:02.0f}.")
    
simulate_all_parameters()   
