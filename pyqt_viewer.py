# Imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import PIL.Image as loadimage
from PIL import ImageOps
from my_mesh.mesh import myMesh
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QWidget, QFrame, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, \
    QScrollArea, QGridLayout, QCheckBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit

from controls.shape_slider import ShapeSlider
from controls.pose_slider import PoseSlider
from controls.trans_slider import TransSlider

from smal_model.smal_torch import SMAL
from pyrenderer import Renderer
# from p3d_renderer import Renderer
from functools import partial
import time

import os
import collections

import scipy.misc
import datetime

import pickle as pkl

NUM_POSE_PARAMS = 35#34
NUM_SHAPE_PARAMS = 41
RENDER_SIZE = 256
DISPLAY_SIZE = 512

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.smal_params = {
            'betas' : torch.zeros(1, NUM_SHAPE_PARAMS),
            'joint_rotations' : torch.zeros(1, NUM_POSE_PARAMS, 3),
            'global_rotation' :  torch.zeros(1, 1, 3),
            'trans' : torch.zeros(1, 1, 3),
        }

        self.model_renderer = Renderer(RENDER_SIZE)
        self.smal_model = SMAL('data/my_smpl_00781_4_all.pkl', 'data\my_smpl_data_00781_4_all.pkl', 'data\symIdx.pkl')

        with open('data/my_smpl_data_00781_4_all.pkl', 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            smal_data = u.load()

        self.toy_betas = smal_data['toys_betas']
        self.bg_image_np = []
        self.setup_ui()


    def get_layout_region(self, control_set):
        layout_region = QVBoxLayout()

        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollAreaWidgetContents = QWidget(scrollArea)
        scrollArea.setWidget(scrollAreaWidgetContents)
        scrollArea.setMinimumWidth(750)

        grid_layout = QGridLayout()

        for idx, (label, com_slider) in enumerate(control_set):
            grid_layout.addWidget(label, idx, 0)
            grid_layout.addWidget(com_slider, idx, 1)
            # grid_layout.addWidget(button, idx, 2)

        scrollAreaWidgetContents.setLayout(grid_layout)

        layout_region.addWidget(scrollArea)
        return layout_region

    def setup_ui(self):
        self.shape_controls = []
        self.pose_controls = []
        self.update_poly = True
        self.toy_pbs = []

        def ctrl_layout_add_separator():
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            ctrl_layout.addWidget(line)

        # SHAPE REGION
        # std_devs = np.std(self.toy_betas, axis=1)###
        std_devs = np.ones_like(np.std(self.toy_betas, axis=1))###

        for idx, toy_std in enumerate(std_devs):
            # label = QLabel("S{0}".format(idx))#
            label = QLabel("PC{0}".format(idx+1))
            sliders = ShapeSlider(idx, toy_std)###
            sliders.value_changed.connect(self.update_model)
            self.shape_controls.append((label, sliders))
            self.toy_pbs.append(QPushButton("T{0}".format(idx)))

        reset_shape_pb = QPushButton('Reset Shape')
        reset_shape_pb.clicked.connect(self.reset_shape)

        self.toy_frame = QFrame()
        self.toy_layout = QGridLayout()
        for idx, pb in enumerate(self.toy_pbs):
            row = idx // 5
            col = idx - row * 5
            pb.clicked.connect(partial(self.make_toy_shape, idx))
            self.toy_layout.addWidget(pb, row, col)

        self.toy_frame.setLayout(self.toy_layout)
        self.toy_frame.setHidden(True)

        # show_toys_cb = QCheckBox('Show Toys', self)
        # show_toys_cb = QCheckBox('Show Pigs', self)
        # show_toys_cb.stateChanged.connect(partial(self.toggle_control, self.toy_frame))
        
        shape_layout = self.get_layout_region(self.shape_controls)
        shape_layout.addWidget(reset_shape_pb)
        # shape_layout.addWidget(show_toys_cb)###

        shape_title = QLabel("Shape Parameter")
        shape_title.setFont(QFont("Roman times", 15, QFont.Bold))

        ctrl_layout = QVBoxLayout()
        ctrl_layout.addWidget(shape_title)
        ctrl_layout.addLayout(shape_layout)
        ctrl_layout.addWidget(self.toy_frame)
        ctrl_layout_add_separator()

        # POSE REGION
        model_joints = NUM_POSE_PARAMS
        for idx in range(model_joints):
            if idx == 0:
                label = QLabel("Root P{0}".format(idx))
                slider = PoseSlider(idx, 2 * np.pi)#, vert_stack = True
                # one_pose_bt = QPushButton(f'Reset P{idx}')
            else:
                label = QLabel("P{0}".format(idx))
                slider = PoseSlider(idx, np.pi)
                # one_pose_bt = QPushButton(f'Reset P{idx}')

            slider.value_changed.connect(self.update_model)
            self.pose_controls.append((label, slider))

        reset_pose_pb = QPushButton('Reset Pose')
        reset_pose_pb.clicked.connect(self.reset_pose)

        root_pose_dict = collections.OrderedDict()
        # root_pose_dict[ "Face Left" ] = np.array([0, 0, np.pi])
        # root_pose_dict[ "Diag Left" ] = np.array([0, 0, 3 * np.pi / 2])
        # root_pose_dict[ "Head On" ] = np.array([0, 0, np.pi / 2])
        # root_pose_dict[ "Diag Right" ] = np.array([0, 0, np.pi / 4])
        # root_pose_dict[ "Face Right" ] = np.array([0, 0, 0])
        # root_pose_dict[ "Straight Up" ] = np.array([np.pi / 2, 0, np.pi / 2])
        # root_pose_dict[ "Straight Down" ] = np.array([-np.pi / 2, 0, np.pi / 2])

        root_pose_dict[ "Face Left" ] = np.array([-np.pi / 2, 0, -np.pi])
        root_pose_dict[ "Diag Left" ] = np.array([-np.pi / 2, 0, -3 * np.pi / 4])
        root_pose_dict[ "Head On" ] = np.array([-np.pi / 2, 0, -np.pi / 2])
        root_pose_dict[ "Diag Right" ] = np.array([-np.pi / 2, 0, -np.pi / 4])
        root_pose_dict[ "Face Right" ] = np.array([-np.pi / 2, 0, 0])

        root_pose_dict[ "Straight Up" ] = np.array([np.pi, np.pi, -np.pi / 2])
        root_pose_dict[ "Straight Down" ] = np.array([np.pi, np.pi, np.pi / 2])

        root_pose_layout = QGridLayout()
        idx = 0
        for key, value in root_pose_dict.items():
            head_on_pb = QPushButton(key)
            head_on_pb.clicked.connect(partial(self.set_known_pose, value))
            root_pose_layout.addWidget(head_on_pb, 0, idx)
            idx = idx + 1
   
        pose_layout = QGridLayout()
        root_pose_layout0 = QGridLayout()

        root_label, root_pose_sliders = self.pose_controls[0]
        pose_title = QLabel("Pose Parameter")
        pose_title.setFont(QFont("Roman times", 15, QFont.Bold))
        pose_layout.addWidget(pose_title, 0, 0)

        root_pose_layout0.addWidget(root_label, 1, 0)
        root_pose_layout0.addWidget(root_pose_sliders, 1, 1)

        pose_layout.addLayout(root_pose_layout0, 1, 0)
        pose_layout.addLayout(self.get_layout_region(self.pose_controls[1:]), 2, 0)
        pose_layout.addWidget(reset_pose_pb)

        ctrl_layout.addLayout(pose_layout)
        ctrl_layout.addLayout(root_pose_layout)
        ctrl_layout_add_separator()

        # TRANSLATION REGION
        trans_label = QLabel("Root Translation".format(idx))
        trans_label.setFont(QFont("Roman times", 15, QFont.Bold))
        self.trans_sliders = TransSlider(idx, -5.0, 5.0, np.array([0.0, 0.0, 0.0]), vert_stack = True)
        self.trans_sliders.value_changed.connect(self.update_model)

        reset_trans_pb = QPushButton('Reset Translation')
        reset_trans_pb.clicked.connect(self.reset_trans)
          
        # Add the translation slider
        trans_layout = QGridLayout()
        trans_layout.addWidget(trans_label, 0, 0)
        trans_layout.addWidget(self.trans_sliders, 1, 0)
        trans_layout.addWidget(reset_trans_pb, 2, 0)

        self.trans_frame = QFrame()
        self.trans_frame.setLayout(trans_layout)
        # self.trans_frame.setHidden(True)
        # show_trans_cb = QCheckBox('Show Translation Parameters', self)
        # show_trans_cb.stateChanged.connect(partial(self.toggle_control, self.trans_frame))

        # ctrl_layout.addWidget(show_trans_cb)
        ctrl_layout.addWidget(self.trans_frame)
        ctrl_layout_add_separator()

        # ACTION BUTTONS
        reset_pb = QPushButton('&All Reset')
        reset_pb.clicked.connect(self.reset_model)

        # export_image_pb = QPushButton('&Export Image')
        # export_image_pb.clicked.connect(self.export_image)

        misc_pbs_layout = QGridLayout() 
        misc_pbs_layout.addWidget(reset_pb, 0, 0)
        # misc_pbs_layout.addWidget(export_image_pb, 0, 1)
        ctrl_layout.addLayout(misc_pbs_layout)
        ctrl_layout_add_separator()

        view_layout = QVBoxLayout()
        view_layout.addStretch(10)
        self.render_img_label = QLabel()
        # 设置阴影 只有加了这步才能设置边框颜色
        # 可选样式有Raised、Sunken、Plain（这个无法设置颜色）等
        self.render_img_label.setFrameShadow(QFrame.Raised)
        self.render_img_label.setStyleSheet(
            'border-width: 1px;border-style: solid;border-color: rgb(255, 170, 0);'
            'background-color: rgb(100, 149, 237);')

        self.render_img_window = QLabel("Image Window")
        self.render_img_window.setFont(QFont("Roman times", 20, QFont.Bold))
        self.render_img_window.setAlignment(QtCore.Qt.AlignCenter)

        view_layout.addWidget(self.render_img_window)
        view_layout.addStretch(0)
        view_layout.addWidget(self.render_img_label)
        view_layout.addStretch(1)
        export_image_pb = QPushButton('&Export 3D Model', minimumHeight=40)
        export_image_pb.clicked.connect(self.export_image)
        import_image_pb = QPushButton('&Import Image', minimumHeight=40)
        import_image_pb.clicked.connect(self.import_image)
        view_layout.addWidget(import_image_pb)
        view_layout.addWidget(export_image_pb)
        view_layout.addStretch(10)

        main_layout = QHBoxLayout()
        main_layout.addLayout(ctrl_layout)
        main_layout.addLayout(view_layout)

        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # WINDOW
        # self.window_title_stem = 'SMAL Model Viewer'
        self.window_title_stem = 'Pig3D Model Viewer'
        self.setWindowTitle(self.window_title_stem)
        
        self.statusBar().showMessage('Ready...')
        self.update_render()
        
        self.showMaximized()


    def update_model(self, value):
        sender = self.sender()

        if type(sender) is ShapeSlider:
            self.smal_params['betas'][0, sender.idx] = value
        elif type(sender) is PoseSlider:
            if sender.idx == 0:
                self.smal_params['global_rotation'][0, sender.idx] = torch.FloatTensor(value)
            else:
                self.smal_params['joint_rotations'][0, sender.idx] = torch.FloatTensor(value)####
        elif type(sender) is TransSlider:
            self.smal_params['trans'][0] = torch.FloatTensor(value)

        if self.update_poly:
            self.update_render()

    def update_render(self, reset=False):
        with torch.no_grad():
            start = time.time()

            verts, joints, Rs, v_shaped = self.smal_model(
                beta=self.smal_params['betas'],
                theta=torch.cat([self.smal_params['global_rotation'], self.smal_params['joint_rotations'][:,1:,:]], dim = 1))

            self.mm = myMesh(v=verts[0], f=self.smal_model.faces)
            if reset:
                print("reset!")
            else:
                import trimesh
                time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                # mesh_posed = trimesh.Trimesh(vertices=verts[0], faces=self.smal_model.faces, process=False)
                # mesh_posed.export('E:/DL/SMALViewer/svm_test/低头/' + time_str + '.obj')

            # normalize by center of mass
            verts = verts - torch.mean(verts, dim = 1, keepdim=True)

            # add on the translation
            verts = verts + self.smal_params['trans']

            end = time.time()
            ellapsed  = end - start
            # print (f"SMAL Time: {ellapsed }")###

            start = time.time()
            rendered_images = self.model_renderer(verts, self.smal_model.faces.unsqueeze(0))
            end = time.time()
            ellapsed = end - start
            # print(f"Renderer Time: {ellapsed }")###

        if self.bg_image_np != []:
            self.image_np = rendered_images[0, :, :, :3] * 0.85 + self.bg_image_np * 0.15
        else:
            self.image_np = rendered_images[0, :, :, :3]
        # plt.Figure()
        # plt.imshow(self.image_np)
        # plt.show()
        self.render_img_label.setPixmap(self.image_to_pixmap(self.image_np, DISPLAY_SIZE))

        self.render_img_label.update()

    def reset_shape(self):
        # Reset sliders to zero
        self.update_poly = False
        for label, com_slider in self.shape_controls:
            com_slider.reset()
        self.update_poly = True
        self.update_render(reset=True)

    def reset_pose(self):
        self.update_poly = False
        for label, com_slider in self.pose_controls[1:]:
            com_slider.reset()
        self.update_poly = True
        self.update_render(reset=True)

    def reset_pose0(self):
        self.update_poly = False
        for label, com_slider in self.pose_controls[:1]:
            com_slider.reset()
        self.update_poly = True
        self.update_render(reset=True)


    def make_toy_shape(self, toy_id):
        self.update_poly = False
        toy_betas = self.toy_betas[toy_id]
        for idx, val in enumerate(toy_betas):
            label, shape_slider = self.shape_controls[idx]
            shape_slider.setValue(val)

        self.update_poly = True
        self.statusBar().showMessage(str(toy_betas))
        self.smal_params['betas'][0] = torch.from_numpy(toy_betas)
        self.update_render()

    def toggle_control(self, layout):
        sender = self.sender()
        layout.setHidden(not sender.isChecked())
            
    def set_known_pose(self, pose):
        label, root_pose_slider = self.pose_controls[0]
        root_pose_slider.setValue(pose)
        root_pose_slider.force_emit()

    def reset_trans(self):
        self.trans_sliders.reset()

    def reset_model(self):
        self.reset_shape()
        self.reset_pose()
        self.reset_trans()
        self.reset_pose0()

    def image_to_pixmap(self, img, img_size):
        im = np.require(img * 255.0, dtype='uint8')
        qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888).copy()
        pixmap = QPixmap(qim)
        return pixmap.scaled(img_size, img_size, QtCore.Qt.KeepAspectRatio)

    def export_image(self):
        out_dir = "output"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        # self.mm.save_ply('E:/DL/SMALViewer/output/' + "{0}.ply".format(time_str))  ##保存ply
        self.mm.save_ply(out_dir + "/{0}.ply".format(time_str))  ##保存ply
        print("Save ply path:", out_dir + "/{0}.ply".format(time_str))
        # scipy.misc.imsave(os.path.join(out_dir, "{0}.png".format(time_str)), self.image_np)

    def import_image(self):
        self.render_img_label.setStyleSheet(
            'border-width: 1px;border-style: solid;border-color: rgb(255, 170, 0);'
            'background-color: rgb(100, 149, 237);')
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp)")
        file_dialog.setViewMode(QFileDialog.Detail)

        if file_dialog.exec_():
            file_name = file_dialog.selectedFiles()[0]
            print("Show image path: ", file_name)
            ori_image = loadimage.open(file_name)
            self.bg_image_np = np.array(ImageOps.pad(ori_image, (256, 256)))/255
            self.image_np = self.image_np[:, :, :3] * 0.85 + self.bg_image_np * 0.15
            self.render_img_label.setPixmap(self.image_to_pixmap(self.image_np, DISPLAY_SIZE))
            self.render_img_label.update()