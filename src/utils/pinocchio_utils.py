import os
import sys
import time
import json
import yaml

import numpy as np
import pinocchio as pin
from urdf_parser_py.urdf import URDF

from math import *

from .robotics_utils import *
from .rotation_utils import *
from .print_utils import *

class PinocchioModel:
    def __init__(self, urdf_dir, T_W0=None, T_EE=None):

        # Import robot

        self._robot_name = os.path.basename(urdf_dir)
        self._robot_type = os.path.basename(os.path.dirname(urdf_dir))

        # Open YAML file
        with open(urdf_dir + "/../robot_configs.yaml".format(self._robot_type)) as yaml_file:
            self._robot_configs = yaml.load(yaml_file, Loader=yaml.FullLoader)

        # Load pinocchio model
        # try:
        #     if self._robot_configs[self._robot_name]["UseDefaultModel"]:
        #         target_robot_name = self._robot_type
        #     else:
        #         target_robot_name = self._robot_name
        # except:
        #     target_robot_name = self._robot_name
        #
        # target_robot_name = self._robot_name

        self.pinModel = pin.buildModelFromUrdf(urdf_dir + "/../{}/model.urdf".format(self._robot_name))
        self.pinData = self.pinModel.createData()
        self.numJoints = self.pinModel.nq

        pin.forwardKinematics(self.pinModel, self.pinData, np.zeros([self.numJoints, 1]))
        pin.updateFramePlacements(self.pinModel, self.pinData)

        self.RobotEEJointIdx = self._robot_configs[self._robot_name]["JointInfo"]["RobotEEJoint"]
        self.RobotMovableJointIdx = self._robot_configs[self._robot_name]["JointInfo"]["RobotMovableJoint"]

        urdf_joints = URDF.from_xml_file(urdf_dir + "/../{}/model.urdf".format(self._robot_name)).joints
        accumulated_num_fixed_joint = np.zeros([len(urdf_joints)], dtype=np.int32)
        for i, joint in enumerate(urdf_joints):
            if joint.type == "fixed":
                accumulated_num_fixed_joint[i:] += 1
        self.pinRobotMovableJointIdx = [idx-accumulated_num_fixed_joint[idx] for idx in self.RobotMovableJointIdx]

        # Robot's base coordinate in world coordinate
        if T_W0 is None:
            self._T_W0 = np.identity(4)
        else:
            self._T_W0 = T_W0
        self._Ad_W0 = Adjoint(self._T_W0)

        if T_EE is None:
            # pinModel = pin.buildModelFromUrdf(urdf_dir + "/../{}/model.urdf".format(self._robot_name))
            # pinData = pinModel.createData()
            # numJoints = pinModel.nq
            #
            # RobotEEJointIdx = self._robot_configs[self._robot_name]["JointInfo"]["RobotEEJoint"]
            #
            # pin.forwardKinematics(pinModel, pinData, np.zeros([numJoints, 1]))
            # pin.updateFramePlacements(pinModel, pinData)
            #
            # # EE joint should be placed in next to the last movable joint !!!
            # T_TCP_target = self.pinData.oMf[2 + 2 * (self.RobotEEJointIdx[0] + 1)].np
            # T_TCP = pinData.oMf[2 + 2 * (RobotEEJointIdx[0] + 1)].np
            #
            # self._T_EE = TransInv(T_TCP_target) @ T_TCP
            #
            # xyz = self._robot_configs[self._robot_name]["EndEffector"]["position"]
            # rpy = self._robot_configs[self._robot_name]["EndEffector"]["orientation"]
            # self._T_EE = self._T_EE @ xyzeul2SE3(xyz, rpy, seq='XYZ', degree=True)
            xyz = self._robot_configs[self._robot_name]["EndEffector"]["position"]
            rpy = self._robot_configs[self._robot_name]["EndEffector"]["orientation"]
            self._T_EE = xyzeul2SE3(xyz, rpy, seq='XYZ', degree=True)

        else:
            self._T_EE = T_EE


    def reset_base(self, T_W0):

        self._T_W0 = T_W0
        self._Ad_W0 = Adjoint(self._T_W0)

    def _single_CLIK(self, T_goal, q, ql, qu):

        eps = 1e-4
        IT_MAX = 200
        DT = 1e-1
        damp = 1e-12

        for _ in range(IT_MAX):
            err = pin.log6(TransInv(self.FK(q)) @ T_goal).np.reshape(-1, 1)

            if np.linalg.norm(err) < eps:
                q_check = np.mod(q + np.pi, 2 * np.pi) - np.pi
                if (True not in (q_check < ql)) and (True not in (q_check > qu)):
                    return q_check
                else:
                    return None

            J = self.Jb(q)
            v = J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = q + v * DT
            # q = pin.integrate(self.pinModel, q, v * DT)

        return None

    def CLIK(self, T_goal, ql, qu, q_init=None, N_trials=10):
        # Closed-Loop Inverse Kinematics
        # q_init: np.ndarray (1d array)
        '''
        : return np.ndarray (1D) radian
        '''

        ql = np.asarray(ql).reshape(-1, 1)
        qu = np.asarray(qu).reshape(-1, 1)
        ql_extend = self._extend_states_expression(ql).reshape(-1)
        qu_extend = self._extend_states_expression(qu).reshape(-1)

        if q_init is None:
            q_init = self._reduce_states_expression(pin.randomConfiguration(self.pinModel, ql_extend, qu_extend))
        else:
            q_init = np.asarray(q_init).reshape(-1, 1)

        for _ in range(N_trials):
            q_res = self._single_CLIK(T_goal, q_init, ql, qu) # 2d array
            if q_res is not None:
                return q_res

            q_init = self._reduce_states_expression(pin.randomConfiguration(self.pinModel, ql_extend, qu_extend))
        return None

    def FK(self, q):
        q_extend = self._extend_states_expression(q)
        return self.computeForwardKinematics(q_extend)

    def Js(self, q):
        q_extend = self._extend_states_expression(q)
        return self.computeSpatialJacobianMatrix(q_extend)[:, self.pinRobotMovableJointIdx]

    def Jb(self, q):
        q_extend = self._extend_states_expression(q)
        return self.computeBodyJacobianMatrix(q_extend)[:, self.pinRobotMovableJointIdx]

    def dJs(self, q, dq):
        q_extend, dq_extend = self._extend_states_expression(q, dq)
        return self.computeTimeDerivativeSpatialJacobianMatrix(q_extend, dq_extend)[:, self.pinRobotMovableJointIdx]

    def dJb(self, q, dq):
        q_extend, dq_extend = self._extend_states_expression(q, dq)
        return self.computeTimeDerivativeBodyJacobianMatrix(q_extend, dq_extend)[:, self.pinRobotMovableJointIdx]

    def M(self, q):
        q_extend = self._extend_states_expression(q)
        return self.computeMassMatrix(q_extend)[:, self.pinRobotMovableJointIdx][self.pinRobotMovableJointIdx, :]

    def Minv(self, q):
        q_extend = self._extend_states_expression(q)
        return self.computeInverseMassMatrix(q_extend)[:, self.pinRobotMovableJointIdx][self.pinRobotMovableJointIdx, :]

    def C(self, q, dq):
        q_extend, dq_extend = self._extend_states_expression(q, dq)
        return self.computeCoriolisMatrix(q_extend, dq_extend)[:, self.pinRobotMovableJointIdx][self.pinRobotMovableJointIdx, :]

    def g(self, q):
        q_extend = self._extend_states_expression(q)
        return self.computeGravityVector(q_extend)[self.pinRobotMovableJointIdx, :]

    def computeForwardKinematics(self, q):
        q = np.asarray(q).reshape(-1, 1)

        pin.forwardKinematics(self.pinModel, self.pinData, q)
        pin.updateFramePlacements(self.pinModel, self.pinData)
        # return self._T_W0 @ self.pinData.oMi[self.numJoints].np @ self._T_EE
        # 2 [ground link + joint] + 2*(TCP index) [robot link + joint]
        # return self._T_W0 @ self.pinData.oMf[2 + 2 * (self.RobotTCPJointIdx[0] + 1)].np @ self._T_EE
        # 2 [ground link + joint] + 2*(Flange index = Last Movable joint idx + 1) [robot link + joint]
        return self._T_W0 @ self.pinData.oMf[2 + 2 * (self.RobotEEJointIdx[-1] + 1)].np @ self._T_EE

    def computeSpatialJacobianMatrix(self, q):
        q = np.asarray(q).reshape(-1, 1)

        pin.forwardKinematics(self.pinModel, self.pinData, q)
        J = pin.computeJointJacobians(self.pinModel, self.pinData)
        return self._Ad_W0 @ J  # [Jv; Jw]

    def computeBodyJacobianMatrix(self, q):
        return AdjointInv(self.computeForwardKinematics(q)) @ self.computeSpatialJacobianMatrix(q)

    def computeTimeDerivativeSpatialJacobianMatrix(self, q, dq):
        q = np.asarray(q).reshape(-1, 1)
        dq = np.asarray(dq).reshape(-1, 1)

        pin.forwardKinematics(self.pinModel, self.pinData, q)
        dJ = pin.computeJointJacobiansTimeVariation(self.pinModel, self.pinData, q, dq)
        return self._Ad_W0 @ dJ

    def computeTimeDerivativeBodyJacobianMatrix(self, q, dq):
        Jb = self.computeBodyJacobianMatrix(q)
        Vb = Jb @ dq
        adj = adjoint(Vb)
        AdjInv = AdjointInv(self.FK(q))
        return (-adj @ AdjInv @ self.computeSpatialJacobianMatrix(q)
                + AdjInv @ self.computeTimeDerivativeSpatialJacobianMatrix(q, dq))

    def computeInverseMassMatrix(self, q):
        return pin.computeMinverse(self.pinModel, self.pinData, np.asarray(q).reshape([-1, 1]))

    def computeMassMatrix(self, q):
        return np.linalg.inv(self.computeInverseMassMatrix(q))

    def computeCoriolisMatrix(self, q, dq):
        return pin.computeCoriolisMatrix(self.pinModel, self.pinData,
                                         np.asarray(q).reshape([-1, 1]), np.asarray(dq).reshape([-1, 1]))

    def computeGravityVector(self, q):
        return pin.computeGeneralizedGravity(self.pinModel, self.pinData,
                                             np.asarray(q).reshape([-1, 1])).reshape([-1, 1])

    def _extend_states_expression(self, q, dq=None, ddq=None):

        if dq is None and ddq is None:
            q_extend = np.zeros([self.numJoints, 1])
            q_extend[self.pinRobotMovableJointIdx, :] = np.asarray(q).reshape(-1, 1)
            return q_extend
        elif ddq is None:
            q_extend = np.zeros([self.numJoints, 1])
            q_extend[self.pinRobotMovableJointIdx, :] = np.asarray(q).reshape(-1, 1)
            dq_extend = np.zeros([self.numJoints, 1])
            dq_extend[self.pinRobotMovableJointIdx, :] = np.asarray(dq).reshape(-1, 1)
            return q_extend, dq_extend
        else:
            q_extend = np.zeros([self.numJoints, 1])
            q_extend[self.pinRobotMovableJointIdx, :] = np.asarray(q).reshape(-1, 1)
            dq_extend = np.zeros([self.numJoints, 1])
            dq_extend[self.pinRobotMovableJointIdx, :] = np.asarray(dq).reshape(-1, 1)
            ddq_extend = np.zeros([self.numJoints, 1])
            ddq_extend[self.pinRobotMovableJointIdx, :] = np.asarray(ddq).reshape(-1, 1)
            return q_extend, dq_extend, ddq_extend

    def _reduce_states_expression(self, q, dq=None, ddq=None):

        if dq is None and ddq is None:
            q_reduce = np.asarray(q).reshape(-1, 1)[self.pinRobotMovableJointIdx, :]
            return q_reduce
        elif ddq is None:
            q_reduce = np.asarray(q).reshape(-1, 1)[self.pinRobotMovableJointIdx, :]
            dq_reduce = np.asarray(dq).reshape(-1, 1)[self.pinRobotMovableJointIdx, :]
            return q_reduce, dq_reduce
        else:
            q_reduce = np.asarray(q).reshape(-1, 1)[self.pinRobotMovableJointIdx, :]
            dq_reduce = np.asarray(dq).reshape(-1, 1)[self.pinRobotMovableJointIdx, :]
            ddq_reduce = np.asarray(ddq).reshape(-1, 1)[self.pinRobotMovableJointIdx, :]
            return q_reduce, dq_reduce, ddq_reduce

