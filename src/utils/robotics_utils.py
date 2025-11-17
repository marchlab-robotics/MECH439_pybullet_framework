import pybullet as p
import numpy as np
from math import *

from .rotation_utils import *


def isNearZero(a, tol=10e-6):
    if np.abs(a) < tol:
        return True
    else:
        return False

def TransInv(T):
    """
    Return the inverse matrix of transformation matrix T

    :param T: Transformation matrix
    :return: Inverse matrix of transformation matrix T
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3:4]

    Tinv = np.identity(4)
    Tinv[0:3, 0:3] = R.T
    Tinv[0:3, 3:4] = -R.T@p
    return Tinv

def VecToso3(V):
    """
    :param V: numpy array [[x],[y],[z]]
    :return V_ceil: [[0,-z,y],[z,0,-x],[-y,x,0]]
    """
    x = V[0, 0]
    y = V[1, 0]
    z = V[2, 0]
    V_ceil = np.array([[0, -z, y],
                       [z, 0, -x],
                       [-y, x, 0]])
    return V_ceil

def so3ToVec(so3):
    """
    :param so3: [[0,-z,y],[z,0,-x],[-y,x,0]]
    :return:
    """
    x = so3[2,1]
    y = so3[0,2]
    z = so3[1,0]
    vec = np.array([[x],[y],[z]])
    return vec

def VecTose3(vec):
    """
    :param vec: [[],[],[],[],[],[]]
    :return:
    """
    omg = vec[0:3, 0:1]
    v = vec[3:6, 0:1]
    omg_ceil = VecToso3(omg)
    top = np.concatenate([omg_ceil, v], 1)
    btm = np.array([[0, 0, 0, 0]])
    se3 = np.concatenate([top, btm], 0)
    return se3

def se3ToVec(se3):
    w = so3ToVec(se3[0:3, 0:3])
    v = se3[0:3, 3:4]
    return np.concatenate([w, v])

def Adjoint(T):
    """
    Return the Adjoint transformation matrix Adj of transformation matrix T

    :param T: Transformation matrix T
    :return: Adjoint transformation matrix Adj
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3:4]
    p_ceil = VecToso3(p)
    p_ceilR = p_ceil@R
    zero = np.zeros((3, 3))
    top = np.concatenate([R, zero], 1)
    btm = np.concatenate([p_ceilR, R], 1)
    Adj = np.concatenate([top, btm], 0)
    
    return Adj

def AdjointInv(T):
    """
    Return the Adjoint transformation matrix Adj of transformation matrix T

    :param T: Transformation matrix T
    :return: Adjoint transformation matrix Adj
    """
    R = T[0:3, 0:3]
    p = T[0:3, [3]]
    p_ceil = VecToso3(p)

    top = np.concatenate([R.T, -R.T @ p_ceil], axis=1)
    btm = np.concatenate([np.zeros([3, 3]), R.T], axis=1)

    return np.concatenate([top, btm], axis=0)

def ad(V):
    pass

def AxisAng3(V):
    pass

def MatrixExp6(M):
    pass

def MatrixLog3(R):
    pass

def MatrixLog6(T):
    pass

def bodyJacobian(Blist, thetalist):
    """
    Body jacobian

    :param Blist: list of body screw
    :param thetalist: list of joint angle
    :return Jb: result SE3
    """
    pass

def FKinSpace(M, Slist, thetalist):
    """
    Forward Kinematics

    :param M: Initial SE3
    :param Slist: list of spatial screw
    :param thetalist: list of joint angle
    :return T: result SE3
    """
    pass
