"""

    PyTorch implementation of the SMAL/SMPL model

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable
import pickle as pkl 
from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation, batch_global_rigid_transformation_biggs,get_beta_scale_mask
from .smal_basics import align_smal_template_to_symmetry_axis, get_smal_template
import torch.nn as nn

# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r

class SMAL(nn.Module):
    def __init__(self, pkl_model_path, pkl_data_path, pkl_id_path, shape_family_id=-1, dtype=torch.float):
        super(SMAL, self).__init__()

        # -- Load SMPL params --
        # with open(pkl_path, 'r') as f:
        #     dd = pkl.load(f)

        self.logscale_part_list = ['legs_l', 'legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l']

        self.betas_scale_mask = get_beta_scale_mask(part_list=self.logscale_part_list)
            
        with open(pkl_model_path, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            dd = u.load()

        # with open(r"E:\DL\SMALViewer\data\zebra_walking_symmetric_pose_prior_with_cov_35parts.pkl", 'rb') as f:
        with open(r"./data/walking_toy_symmetric_pose_prior_with_cov_35parts.pkl", 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            dd_pose = u.load()

        self.pic = torch.tensor(np.array(dd_pose['pic'].r), dtype=torch.float32)
        self.mean_pose = torch.tensor(dd_pose['mean_pose'], dtype=torch.float32)#平均姿态
        self.cov = torch.tensor(dd_pose['cov'], dtype=torch.float32)#协方差矩阵

        self.f = dd['f']

        self.faces = torch.from_numpy(self.f.astype(int))

        v_template = get_smal_template(model_name=pkl_model_path, data_name=pkl_data_path, shape_family_id = shape_family_id)
        v, self.left_inds, self.right_inds, self.center_inds = \
            align_smal_template_to_symmetry_axis(v_template, sym_file = pkl_id_path)

        # Mean template vertices
        self.v_template = Variable(
            torch.Tensor(v),
            requires_grad=False)
        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis
        
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T

        self.shapedirs = Variable(torch.Tensor(shapedir), requires_grad=False)

        # Regressor for joint locations given shape 
        '''dog'''
        # self.J_regressor = Variable(torch.Tensor(dd['J_regressor'].T.todense()), requires_grad=False)

        '''pig'''
        self.J_regressor = Variable(torch.Tensor(dd['J_regressor'].T),requires_grad=False)

        # Pose blend shape basis
        num_pose_basis = dd['posedirs'].shape[-1]
        
        posedirs = np.reshape(undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T

        self.posedirs = Variable(torch.Tensor(posedirs), requires_grad=False)

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)
        self.kintree_table = dd['kintree_table'].astype(np.int32)

        # LBS weights
        self.weights = Variable(torch.Tensor(undo_chumpy(dd['weights'])), requires_grad=False)

    def __call__(self, beta, theta, trans=None, pose=None, del_v=None, betas_logscale=None, get_skin=True):

        # print("theta: ",theta[0][:3,:])
        if True:
            nBetas = beta.shape[1]
        else:
            nBetas = 0


        # NEW: allow that rotation is given as rotation matrices instead of axis angle rotation
        #   theta: BSxNJointsx3 or BSx(NJoints*3)
        #   pose: NxNJointsx3x3
        # if (theta is None) and (pose is None):
        #     raise ValueError("Either pose (rotation matrices NxNJointsx3x3) or theta (axis angle BSxNJointsx3) must be given")
        # elif (theta is not None) and (pose is not None):
        #     raise ValueError("Not both pose (rotation matrices NxNJointsx3x3) and theta (axis angle BSxNJointsx3) can be given")

        
        # v_template = self.v_template.unsqueeze(0).expand(beta.shape[0], 3889, 3)
        v_template = self.v_template
        # 1. Add shape blend shapes

        # print(f"其他动物 beta:{beta.shape}{beta}")
        # print(f"姿态变化 theta:{theta.shape}{theta}")
        
        if nBetas > 0:
            if del_v is None:
                # print("bata: ", beta)
                v_shaped = v_template + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
            else:
                v_shaped = v_template + del_v + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
        else:
            if del_v is None:
                v_shaped = v_template.unsqueeze(0)
            else:
                v_shaped = v_template + del_v


        # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)
        

        import json
        import numpy as np
        from .geometry_utils import rot6d_to_rotmat, rotmat_to_rot6d
        from svm_test.cal_head import rot6D_to_degree, rotation_matrix_to_axis_angle,rotation_matrix_to_angles
        with open(r"./data/pose.json", 'r') as f:
            pose_prior = json.load(f)
            pose_prior = np.asarray(pose_prior)
            # pose_prior = rot6d_to_rotmat(torch.tensor(pose_prior))
            #
            # print("********************************************")
            # print(batch_rodrigues(rotation_matrix_to_axis_angle(pose_prior[15])))
            # print(batch_rodrigues(rotation_matrix_to_axis_angle(pose_prior[16])))
            # print("********************************************")
            # pose_prior = torch.tensor(pose_prior[:, :, :2].reshape((-1, 6)))
            # pose_prior = torch.as_tensor(pose_prior)[0]
            # print(f"pose_prior:{pose_prior.shape}\n", pose_prior[7, ::])
        # theta_pose = rotmat_to_rot6d(torch.reshape(batch_rodrigues(torch.reshape(theta, [-1, 3])), [-1, 35, 3, 3]))
        # print(f"theta_pose:{theta_pose.shape}\n", theta_pose[7, ::])

        # mat_pose = rot6d_to_rotmat(pose_prior[15,:])
        # from scipy.spatial.transform import Rotation
        # euler_angles = Rotation.from_matrix(mat_pose)
        # print(euler_angles.as_euler('xyz'))

        # pose_prior[6,:] = theta_pose[6, :]
        # pose_prior[8, :] = pose_prior[7, :]###
        # pose = rot6d_to_rotmat(pose_prior)
        # pose = torch.tensor(pose[None, ::], dtype=torch.float32)
        # pose[0, 0, ::] = torch.reshape( batch_rodrigues(torch.reshape(theta, [-1, 3])), [-1, 35, 3, 3])[0, 0, ::]

        # 3. Add pose blend shapes
        # N x 24 x 3 x 3
        if pose is None:
            Rs = torch.reshape( batch_rodrigues(torch.reshape(theta, [-1, 3])), [-1, 35, 3, 3])
        else:
            Rs = pose
            # new_pose = torch.reshape(theta, [-1]) - self.mean_pose
            # new_pose = torch.matmul(new_pose, self.cov*2)
            # new_pose[:3] = theta[0][0]
            #
            # # Rs = torch.reshape(batch_rodrigues(torch.reshape(torch.cat((theta[0][0], self.pic[index])), [-1, 3])), [-1, 35, 3, 3])
            # Rs = torch.reshape(batch_rodrigues(torch.reshape(new_pose, [-1, 3])), [-1, 35, 3, 3])

        import numpy as np #静态
        # import copy
        # for K in Rs[0]:
        #     K_1 = torch.where(torch.abs(K) == 1.0)
        #     # print("K_1:",K_1)
        #     if K_1[0].shape[0] == 3:
        #         break
        #     else:
        #         for I in zip(K_1[0], K_1[1]):
        #             # print('OLD:',K)
        #             what = copy.deepcopy(K[I[0],I[1]])
        #             K[I[0], :][torch.where(K[I[0], :]!= what)] = 0
        #             K[:, I[1]][torch.where(K[:, I[1]]!= what)] = 0
                    # print("NEW:", K)

        # print("=========================================\n",Rs[0])

        # Rs = torch.Tensor(np.array(list(np.eye(3,3))*35).reshape(1,35,3,3))
        
        # Ignore global rotation.
        pose_feature = torch.reshape(Rs[:, 1:, :, :] - torch.eye(3).to(beta.device), [-1, 306])#

        '''++++add++++'''
        # x_legs = Rs[:, 1:, :, :].reshape((-1, 3, 3)) @ torch.tensor(np.array([[0], [0], [-1]]), dtype=torch.float32)
        x_legs = Rs[:, 1:, :, :].reshape((-1, 3, 3)) @ torch.tensor(np.array([[0], [-1], [0]]), dtype=torch.float32)
        # print('poselegssidemovement:', (x_legs[:, 0]**2).mean())
        
        v_posed = torch.reshape( torch.matmul(pose_feature, self.posedirs), [-1, self.size[0], self.size[1]]) + v_shaped


        '''new: add corrections of bone lengths to the template  (before hypothetical pose blend shapes!)
        see biggs batch_lbs.py'''
        #@运算，矩阵乘法
        betas_logscale = torch.tensor([-0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        betas_scale = torch.exp(betas_logscale @ self.betas_scale_mask.to(betas_logscale.device))
        scaling_factors = betas_scale.reshape(-1, 35, 3)
        scale_factors_3x3 = torch.diag_embed(scaling_factors, dim1=-2, dim2=-1)#对角元素输出对角矩阵

        # 4. Get the global joint location
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents)####important,origin
        # self.J_transformed, A = batch_global_rigid_transformation_biggs(Rs, J, self.parents,
        #                                     scale_factors_3x3, betas_logscale=betas_logscale)###

        # import matplotlib.pyplot as plt
        # import numpy as np
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure(figsize=(10, 8))
        # ax = Axes3D(fig)
        # X = A[0,:,:3,3][:, 0]
        # Y = A[0,:,:3,3][:, 1]
        # Z = A[0,:,:3,3][:, 2]
        # ax.scatter3D(X, Y, Z, c="r", s=100)
        # plt.show()

        # if torch.any(torch.abs(A) > 1) :
        #     A[torch.where(A > 1)] = 1
        #     A[torch.where(A < -1)] = -1

        # print("A:", torch.where(torch.abs(A) > 1))

        # 5. Do skinning:
        num_batch = theta.shape[0]
        
        weights_t = self.weights.repeat([num_batch, 1])
        W = torch.reshape(weights_t, [num_batch, -1, 35])

        # if torch.any(torch.abs(W) > 1) :
        #     W[torch.where(W > 1)] = 1
        #     W[torch.where(W < -1)] = -1

        # colors = np.zeros_like(weights_t[:, 0])
        # import matplotlib.pyplot as plt
        # import numpy as np
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure(figsize=(16, 14))
        # ax = Axes3D(fig)
        # plt.ion()
        # for i in range(35):
        #     shapes = np.ones_like(weights_t[:, 0])
        #     vert_weigths = weights_t[:, i]
        #     # colors[vert_weigths != 0] = i
        #     shapes = shapes + np.array(vert_weigths) * 100
        #     ax.clear()
        #     ax.scatter3D(v_posed[0][:,0], v_posed[0][:,1], v_posed[0][:,2], c=colors, s=shapes)
        #     ax.set_xlim(-1.0, 1.0)
        #     ax.set_ylim(-1.0, 1.0)
        #     ax.set_zlim(-0.5, 0.5)
        #     ax.text3D(0, 0, 1, f"{i}", transform=ax.transAxes)
        #     plt.pause(2)
        # plt.ioff()


        # A[0][:,0:4,3] = 0 #去掉平移 少了蒙皮约束
        T = torch.reshape(torch.matmul(W, torch.reshape(A, [num_batch, 35, 16])), [num_batch, -1, 4, 4])
        # print("T.shape: ", T.shape)
        '''约束'''
        import numpy as np
        # T[torch.where(T > 1)] = 1.0
        # T[torch.where(T < -1)] = -1.0

        v_posed_homo = torch.cat([v_posed, torch.ones([num_batch, v_posed.shape[1], 1]).to(device=beta.device)], 2)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

        verts = v_homo[:, :, :3, 0]

        # verts = v_posed_homo[:, :, :3]###

        # import matplotlib.pyplot as plt
        # import numpy as np
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure(figsize=(10, 8))
        # ax = Axes3D(fig)
        # xlist1 = np.array(verts[0])[:, 0]
        # ylist1 = np.array(verts[0])[:, 1]
        # zlist1 = np.array(verts[0])[:, 2]
        # ax.scatter3D(xlist1, ylist1, zlist1)
        # plt.show()

        # shape = np.ones(3889)
        # for x in range(verts.shape[1]):
        #     gap = np.array(v_homo[0, x, :3, 0]) - np.array(v_posed_homo[0, x, :3])
        #     shape[x] = shape[x] + sum(gap)*1000
        # ax.scatter3D(xlist1, ylist1, zlist1, c="blue", s=shape)
        # # x, y = np.where(np.array(v_homo[0, :, :3, 0]) - np.array(v_posed_homo[0, :, :3]) > 0.0)
        # # for i, j in zip(x, y):
        # #     ax.text(verts[0, i][0],verts[0, i][1],verts[0, i][2], 'F')
        # ax.set_xlim(-0.5, 0.5)
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(-0.5, 0.5)
        # plt.show()


        if trans is None:
            trans = torch.zeros((num_batch, 3)).to(device=beta.device)

        verts = verts + trans[:, None, :]

        # Get joints:
        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
        # joints = torch.stack([joint_x, joint_y, joint_z], dim=2)#
        joints = self.J_transformed
        # joint_x = joints[:,:,0]
        # joint_y = joints[:,:,1]
        # joint_z = joints[:,:,2]


        jot = joints[:,16,:] - verts[0][257]
        p1 = verts[0][1658]#腿
        p2 = verts[0][3221]
        p3 = verts[0][1341]
        p4 = verts[0][3538]
        # p1 = verts[0][267]
        # p2 = verts[0][375]
        # p1 = joints[0, 6, :]#上半身中轴
        # p2 = joints[0, 15, :]
        # p4 = verts[0][192]
        v1 = np.array(p2 - p1)
        v2 = np.array(p4 - p1)
        n = np.cross(v1, v2)
        g = n / np.linalg.norm(n)
        print(g)
        # 计算向量的点积
        dot_product = np.dot(jot, g)
        # 计算向量的模长
        h_norm = np.linalg.norm(jot)
        g_norm = np.linalg.norm(g)
        # 计算夹角
        angle = np.arccos(dot_product / (h_norm * g_norm)) * 180 / np.pi
        if angle < 85:
            print("猪头低头")
        elif angle > 95:
            print("猪头抬头")
        else:
            print("猪头正对前方")


        # jot2 = joints[0][2] - joints[0][1]
        # jot3 = joints[0][3] - joints[0][2]
        # jot4 = joints[0][4] - joints[0][3]
        # print(jot2, jot3, jot4)

        # from svm_test.cal_head import align_triangle_with_z_axis
        # new_jot = align_triangle_with_z_axis([joints[0][6], joints[0][15], joints[0][16]])
        # print([joints[0][6], joints[0][15], joints[0][16]])
        # print(new_jot)
        # jot6 = new_jot[0] - new_jot[1]
        # jot15 = new_jot[1]
        # jot16 = new_jot[2] - new_jot[1]

        # jot6 = joints[0][6] - joints[0][15]
        # jot16 = joints[0][16] - joints[0][15]
        # jot15 = joints[0][15]
        # cos_theta = np.dot(jot6, jot16) / (np.linalg.norm(jot6) * np.linalg.norm(jot16))
        # deg = np.rad2deg(np.arccos(cos_theta))
        # print(deg,"度")
        # print("jot16: ",jot16)
        # print("jot16: ",joints[0][16] - joints[0][15])


        # print(jot16-jot15)
        # jj = jot16-jot15
        # if jj[0]<-0.1:
        #     print("左转头")
        # if jj[0]>0.1:
        #     print("右转头")
        # if jj[1]<-0.1:
        #     print("低头")
        # if jj[1]>0.1:
        #     print("抬头")

        from svm_test.cal_head import roll_and_pitch_and_yaw,calc_euler_angles,rotate_to_direction,rotation_matrix_to_euler_angles,rotation_matrix_to_angles
        # v_v = [267, 370, 257]
        # jot6 = verts[0][v_v[0]]
        # jot15 = verts[0][v_v[1]]
        # jot16 = verts[0][v_v[2]] - jot15
        # v = jot15 - jot6  #方向
        # jot16_0, R = rotate_to_direction(jot16, v)
        # eu_an = rotation_matrix_to_euler_angles(R)
        # print("ori:", jot15, jot16_0)
        # print("new:", jot16)
        # print(eu_an)
        # print(torch.cross(jot16,jot16_0),torch.norm(torch.cross(jot16,jot16_0)))


        # from numpy.linalg import solve
        # '''z轴'''
        # a = np.mat([[jot16_0[0], -jot16_0[1]], [jot16_0[1], jot16_0[0]]])
        # b = np.mat([jot16[0], jot16[1]]).T
        # z = solve(a,b)
        # '''y轴'''
        # a = np.mat([[jot16_0[0], jot16_0[2]], [jot16_0[2], -jot16_0[0]]])
        # b = np.mat([jot16[0], jot16[2]]).T
        # y = solve(a,b)
        # '''x轴'''
        # a = np.mat([[jot16_0[1], -jot16_0[2]], [jot16_0[2], jot16_0[1]]])
        # b = np.mat([jot16[1], jot16[2]]).T
        # x = solve(a,b)
        # roll = np.arccos(z[0])
        # yaw = np.arccos(y[0])
        # pitch = np.arccos(x[0])
        #
        # print(roll, yaw, pitch)

        # if (jot1[1][1]-jot1[0][1])>0.1 or (jot1[2][1] - jot1[1][1])>0.1:
        #     print("抬头")
        # if (jot1[1][1]-jot1[0][1])<0.1 or (jot1[2][1] - jot1[1][1])<0.1:
        #     print("低头")
        # if abs(jot1[1][2] - jot1[0][2]) > 0.1 or abs(jot1[2][2] - jot1[1][2]) > 0.1:
        #     print("转头")

        # print("ori:", calc_euler_angles(jot0))
        # print("new:", calc_euler_angles(jot1))
        print("==================================")


        # print("joints", joints)
        # joints = self.J_transformed

        # joints = torch.cat([
        #     joints,
        #     verts[:, None, 1863], # end_of_nose
        #     verts[:, None, 26], # chin
        #     verts[:, None, 2124], # right ear tip
        #     verts[:, None, 150], # left ear tip
        #     verts[:, None, 3055], # left eye
        #     verts[:, None, 1097], # right eye
        #     ], dim = 1)

        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.figure(figsize=[10, 8])
        # ax = plt.axes(projection="3d")
        # joints_x, joints_y, joints_z = joints.cpu()[0][:, 0].detach().numpy(), joints.cpu()[0][:, 1].detach().numpy(), joints.cpu()[0][:,2].detach().numpy()
        # verts_x, verts_y, verts_z = verts.cpu()[0][:, 0].cpu().detach().numpy(), verts.cpu()[0][:, 1].cpu().detach().numpy(),verts.cpu()[0][:, 2].cpu().detach().numpy()
        #
        # # ax.scatter3D(joints_x, joints_y, joints_z, s=50, c='red', label='3d')
        # ax.scatter3D(verts_x, verts_y, verts_z, s=10, c='blue', label='3d',alpha=0.5)##表面的点
        # for i, j in enumerate(config.PIG_MODEL_JOINTS_NAME):
        #     ax.text3D(joints.cpu()[0][i][0].detach().numpy(), joints.cpu()[0][i][1].detach().numpy(),
        #               joints.cpu()[0][i][2].detach().numpy(), j)
        # ax.scatter3D(proj_points[0][:, 0].detach().numpy(), proj_points[0][:, 1].detach().numpy(),
        #              np.zeros_like(proj_points[0][:,0].detach().numpy()), s=50, c='blue', label='2d')
        # ax.legend()
        # kintree_table = [[6, 7], [7, 8], [8, 11], [9, 10], [10, 11], [3, 4], [4, 5], [3, 4],
        #                  [0, 1], [1, 2], [2, 5], [2, 8], [2, 15],
        #                  [8, 16], [15, 19], [16, 20],
        #                  [16, 22], [15, 21], [21, 17], [22, 17], [18, 17],
        #                  [11, 12], [5, 12], [12, 13], [13, 14]]
        # for i in self.kintree_table.T:
        #     if i[0] < 0:
        #         i = [0,0]
        #     x1, y1, z1 = [], [], []
        #     x2, y2, z2 = [], [], []
        #     for j in i:  # 两个点相连
        #         x1.append(float(joint_x[0][j]))
        #         y1.append(float(joint_y[0][j]))
        #         z1.append(float(joint_z[0][j]))
        #         x2.append(float(Jx[0][j]))
        #         y2.append(float(Jy[0][j]))
        #         z2.append(float(Jz[0][j]))
        #     ax.plot3D(x1, y1, z1, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=10,
        #               label="first")
        #     ax.plot3D(x2, y2, z2, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=10, label="second")
        # ax.text3D(x1[0], y1[0], z1[0], "3d", fontsize=10)
        # # ax.text3D(x2[0], y2[0], z2[0], "second", fontsize=10)
        # # plt.savefig(rf"E:\DL\SMALify\outputs\pigs\vis_joints\{time.time()}.png")
        # plt.pause(5)
        # plt.close('all')
        # plt.xlabel('X')
        # plt.ylabel('Y')  # y 轴名称旋转 38 度
        # ax.set_zlabel('Z')  # 因为 plt 不能设置 z 轴坐标轴名称，所以这里只能用 ax 轴来设置（当然，x 轴和 y 轴的坐标轴名称也可以用 ax 设置）
        # ax.set_xlim(-1.0, 1.0)
        # ax.set_ylim(-0.5, 0.5)
        # ax.set_zlim(-0.5, 0.5)
        #
        # import time
        # time0 = time.time()
        # # plt.savefig(f"/media/scau2311/A/xcg/SMALify/outputs/pigs/000000054901/vis_results/3d_joint_{time0}.jpg")

        # plt.show()


        if get_skin:
            return verts, joints, Rs, v_shaped
        else:
            return joints











