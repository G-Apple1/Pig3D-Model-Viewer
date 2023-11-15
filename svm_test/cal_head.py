import glob
import os
import math
import pickle as pkl
from plyfile import PlyData, PlyElement
import numpy as np
import trimesh



def pig_3d_mesh():
    pkl_model_path = r"E:\DL\SMALViewer\data\my_smpl_00781_4_all.pkl"
    with open(pkl_model_path, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        dd = u.load()
    return dd

def dot_and_cross(x):#[[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]
    z1 = np.array([x[1][0] - x[0][0], x[1][1] - x[0][1], x[1][2] - x[0][2]])
    z2 = np.array([x[2][0] - x[1][0], x[2][1] - x[1][1], x[2][2] - x[1][2]])
    Lz1 = np.sqrt(z1.dot(z1))
    Lz2 = np.sqrt(z2.dot(z2))
    angle = np.arccos(np.dot(z1, z2)/(Lz1*Lz2))
    return np.insert(np.cross(z1, z2), 3, angle)

def roll_and_pitch_and_yaw(x):
    v1 = np.array([x[1][0] - x[0][0], x[1][1] - x[0][1], x[1][2] - x[0][2]])
    v2 = np.array([x[2][0] - x[1][0], x[2][1] - x[1][1], x[2][2] - x[1][2]])
    # 计算法向量n
    n = np.cross(v1, v2)
    # 计算滚轮角roll
    sin_roll = n[0]
    cos_roll = np.sqrt(n[1] ** 2 + n[2] ** 2)
    roll = np.arctan2(sin_roll, cos_roll)
    # 计算俯仰角pitch
    cos_pitch = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
    sin_pitch = v1[2]
    pitch = np.arctan2(sin_pitch, cos_pitch)
    # 计算偏转角yaw
    sin_yaw = -v1[0] * n[1] + v1[1] * n[0]
    cos_yaw = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    yaw = np.arctan2(sin_yaw, cos_yaw)
    # 将弧度转换为角度
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    # return [roll_deg, pitch_deg, yaw_deg]
    return [roll, pitch, yaw]

def load_from_ply(filename):
    plydata = PlyData.read(filename)
    plydata = plydata
    f = np.vstack(plydata['face'].data['vertex_indices'])
    x = plydata['vertex'].data['x']
    y = plydata['vertex'].data['y']
    z = plydata['vertex'].data['z']
    v = np.zeros([x.size, 3])
    v[:, 0] = x
    v[:, 1] = y
    v[:, 2] = z
    return v #x, y, z


import numpy as np

def align_triangle_with_z_axis(vertices):
    p1 = np.array(vertices[0])
    p2 = np.array(vertices[1])
    p3 = np.array(vertices[2])

    # 计算三角面的法向量
    normal = np.cross(p2 - p1, p3 - p1)

    # 计算法向量与z轴的夹角
    theta = np.arctan2(np.sqrt(normal[0] ** 2 + normal[1] ** 2), normal[2])

    # 将三角面绕y轴旋转相应角度使法向量与z轴平行
    rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                                [0, 1, 0],
                                [-np.sin(theta), 0, np.cos(theta)]])
    p1_new = np.dot(rotation_matrix, p1)
    p2_new = np.dot(rotation_matrix, p2)
    p3_new = np.dot(rotation_matrix, p3)
    return p1_new,p2_new,p3_new

from scipy.spatial.transform import Rotation as R
import torch
from scipy.spatial.transform import Rotation

def rotation_matrix_to_axis_angle(R):
    r = Rotation.from_matrix(R)
    axis_angle = r.as_rotvec()
    angle = np.linalg.norm(axis_angle)
    axis = axis_angle / angle if angle != 0 else np.array([1, 0, 0])
    # return axis, angle
    return torch.tensor(axis * angle)


def rot6D_to_degree(R_6d):
    # 输入一个6D旋转矩阵
    # 从旋转矩阵中提取欧拉角
    r = R.from_matrix(np.array(R_6d))
    euler = r.as_euler('xyz', degrees=True)
    # 输出欧拉角
    return euler

def calc_euler_angles(x):
    vec1 = np.array([x[1][0] - x[0][0], x[1][1] - x[0][1], x[1][2] - x[0][2]])
    vec2 = np.array([x[2][0] - x[0][0], x[2][1] - x[0][1], x[2][2] - x[0][2]])
    # 计算夹角
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    theta = np.arccos(cos_theta)

    # 计算旋转轴
    n = np.cross(vec1, vec2)
    n /= np.linalg.norm(n)

    # 确定旋转方向
    if n[2] > 0:
        theta = -theta

    # 计算旋转矩阵
    c = np.cos(theta)
    s = np.sin(theta)
    v = 1 - c
    mat = np.array([[n[0] ** 2 * v + c, n[0] * n[1] * v - n[2] * s, n[0] * n[2] * v + n[1] * s],
                    [n[0] * n[1] * v + n[2] * s, n[1] ** 2 * v + c, n[1] * n[2] * v - n[0] * s],
                    [n[0] * n[2] * v - n[1] * s, n[1] * n[2] * v + n[0] * s, n[2] ** 2 * v + c]])

    # 计算欧拉角
    if n[2] == 1:
        # 特殊情况1：n=[0,0,1]，滚转角为0
        pitch = np.arctan2(mat[2, 0], np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2))
        yaw = 0
        roll = 0
    elif n[2] == -1:
        # 特殊情况2：n=[0,0,-1]，滚转角为0
        pitch = np.arctan2(mat[2, 0], np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2))
        yaw = np.pi
        roll = 0
    else:
        # 一般情况
        pitch = np.arctan2(-mat[2, 0], np.sqrt(1 - mat[2, 0] ** 2))
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
        roll = np.arctan2(mat[2, 1], mat[2, 2])

    # 将角度转换为弧度
    # pitch, yaw, roll = np.deg2rad(pitch), np.deg2rad(yaw), np.deg2rad(roll)

    # 返回俯仰角、偏转角、滚转角
    # return np.array([pitch, yaw, roll])

    # 返回欧拉角
    return [pitch, yaw, roll]

import numpy as np

def rotate_to_direction(v1, v2):
    # 计算单位向量和夹角
    v1_hat = v1 / np.linalg.norm(v1)
    v2_hat = v2 / np.linalg.norm(v2)
    cos_theta = np.dot(v1_hat, v2_hat)
    theta = np.arccos(cos_theta)

    # 计算旋转轴
    r = np.cross(v1_hat, v2_hat)

    # 计算旋转矩阵
    rx, ry, rz = r
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    R = cos_theta * np.eye(3) + (1 - cos_theta) * np.outer(r, r) + sin_theta * np.array([
        [0, -rz, ry],
        [rz, 0, -rx],
        [-ry, rx, 0]
    ])

    # 计算旋转后的向量
    v1_prime = np.dot(R, v1)

    return v1_prime,R

def rotation_matrix_to_euler_angles(R):
    # 计算欧拉角
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy < 1e-6:
        # 无法计算欧拉角，设定为默认值
        x = 0
        y = np.arctan2(-R[2, 0], R[0, 0])
        z = np.arctan2(-R[1, 2], R[1, 1])
    else:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])

    # 将弧度转换为角度
    x = np.rad2deg(x)
    y = np.rad2deg(y)
    z = np.rad2deg(z)

    return [x, y, z]

def rotation_matrix_to_angles(R):
    # 计算绕x轴的旋转角度
    theta_x = np.arctan2(R[2, 1], R[2, 2])

    # 计算绕y轴的旋转角度
    c2 = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if c2 != 0:
        theta_y = np.arctan2(-R[2, 0], c2)
    else:
        theta_y = np.arctan2(-R[2, 0], 0)

    # 计算绕z轴的旋转角度
    theta_z = np.arctan2(R[1, 0], R[0, 0])

    # 将弧度转换为角度
    theta_x = np.rad2deg(theta_x)
    theta_y = np.rad2deg(theta_y)
    theta_z = np.rad2deg(theta_z)

    return [theta_x, theta_y, theta_z]

'''face_keypoints: 257 nose, 237 chin, 3700 lelft ear tip, 1820 right ear tip, 3816 left eye, 1936 right eye, 321 throat'''
'''joints: 6 胸口, 15 脖子, 16 头, 34 右耳, 33 左耳, 32 嘴巴'''
def load_dataset(data_type):
    dd = pig_3d_mesh()
    all_data = []
    all_labels = []
    head_j = [[6, 15, 16],[6, 15, 32],[6, 15, 33],[6, 15, 34],[15, 16, 32],[15, 16, 33], [15, 16, 34]]
    head_v = [[267, 370, 257]]#,[370, 235, 257]

    for q, k in enumerate(["低头", "抬头", "转头"]):#
        for i in glob.glob(f"E:/DL/SMALViewer/svm_test/dataset/{data_type}/{k}/*"):
            mesh = trimesh.load(i) # 加载OBJ模型文件
            verts = mesh.vertices# 获取顶点坐标
            head_ang = []
            for w in head_v:
                jot6 = verts[w[0]]
                jot15 = verts[w[1]]
                jot16 = verts[w[2]] #- jot15
                new_dist = np.sqrt(np.sum((jot6 - jot16)**2))
                old_dist = np.sqrt(np.sum((jot15-jot6)**2)) + np.sqrt(np.sum((jot16-jot15)**2))
                # v = jot15 - jot6  # 方向
                # jot16_0, R = rotate_to_direction(jot16, v)
                # eu_an = rotation_matrix_to_euler_angles(R)
                # jot = [[joint_x[i], joint_y[i], joint_z[i]] for i in w]
                # euclidean_dist = np.sqrt(np.sum((joints[w[0]] - joints[w[-1]])**2))
                # # manhattan_dist = np.sum(np.abs(w[0] - w[-1]))
                # dot_cross = dot_and_cross(jot)
                # roll_pitch_yaw = roll_and_pitch_and_yaw(jot)
                # # head_ang = head_ang + dot_cross.tolist() + roll_pitch_yaw + [euclidean_dist]
                head_ang = head_ang + eu_an
            print(data_type, k, head_ang)
            all_data.append(head_ang)
            all_labels.append(q)
    return all_data, all_labels

# import shutil
# dst_dir = r"F:\Postgraduate_time\My_research2\BARC实验\backbone_old\resnet_fc512_trf\stanext24_pig_val_e500"
# test_dir = "E:/DL/SMALViewer/svm_test/image"
# for i in os.listdir(test_dir):
#     for j in glob.glob(os.path.join(test_dir, i)+"/*.png"):
#         name = os.path.basename(j)[:-4]
#         if os.path.exists(os.path.join(dst_dir, f"mesh_posed_{name.split('_')[-1]}.obj")):
#             shutil.copy(os.path.join(dst_dir, f"mesh_posed_{name.split('_')[-1]}.obj"), f"E:/DL/SMALViewer/svm_test/dataset/test/{i}/mesh_posed_{name}.obj")
#         elif os.path.exists(os.path.join(dst_dir, f"mesh_posed_pig_{name.split('_')[-1]}.obj")):
#             shutil.copy(os.path.join(dst_dir, f"mesh_posed_pig_{name.split('_')[-1]}.obj"),
#                         f"E:/DL/SMALViewer/svm_test/dataset/test/{i}/mesh_posed_{name}.obj")
#         else:
#             print(os.path.join(dst_dir, f"mesh_posed_{name.split('_')[-1]}.obj"))




from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score

def main():
    # 训练数据集
    X,Y = load_dataset("train")
    x_train, _, y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=100)#, shuffle=True
    # # 测试数据集
    # # _,x_test,_, y_test = train_test_split(load_dataset("test")[0],load_dataset("test")[1], test_size=1)
    x_test, y_test = load_dataset("test")
    # 创建SVM分类器
    clf = svm.SVC(kernel="poly")#, C=2, gamma=8, degree=10
    # 定义参数网格
    param_grid = [
        {'kernel': ['poly','linear'], 'C': list(range(1, 11, 2)), 'gamma': list(range(1, 11, 2))},#, 'degree':list(range(3, 11, 1))
    ]
    '''进行网格搜索'''
    # svc = svm.SVC()
    # grid_search = GridSearchCV(svc, param_grid, cv=5)
    # grid_search.fit(x_train, y_train)
    # print("Best Parameters:", grid_search.best_params_)# 输出最佳参数组合
    # print("Best Score:", grid_search.best_score_)# 输出最佳分数

    '''使用训练好的SVM分类器进行预测'''
    clf.fit(x_train, y_train)# 训练SVM分类器
    y_pred = clf.predict(x_test)
    # acc = np.sum(y_pred == y_test) / y_test.shape[0]
    acu_train = clf.score(x_train, y_train)
    acu_test = clf.score(x_test, y_test)
    recall = recall_score(y_test, y_pred, average="macro")
    print(y_test)
    print(y_pred)
    print("预测的精度：", acu_train, acu_test, recall)# 输出预测结果

if __name__ == "__main__":
    main()




