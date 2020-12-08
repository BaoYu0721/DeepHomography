import numpy as np
import os
import cv2
import torch
from dataset import make_mesh
from torch_homography_model import build_model

class SingleImgLoader:
    def __init__(self, patch_w=560, patch_h=315, rho=16, WIDTH=640, HEIGHT=360):
        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.rho = rho
        self.x_mesh, self.y_mesh = make_mesh(self.patch_w, self.patch_h)

    def getData(self, path1, path2):
        img_1 = cv2.imread(path1)
        img_2 = cv2.imread(path2)

        height, width = img_1.shape[:2]
        if height != self.HEIGHT or width != self.WIDTH:
            img_1 = cv2.resize(img_1, (self.WIDTH, self.HEIGHT))
        img_1 = (img_1 - self.mean_I) / self.std_I
        img_1 = np.mean(img_1, axis=2, keepdims=True)
        img_1 = np.transpose(img_1, [2, 0, 1])

        height, width = img_2.shape[:2]
        if height != self.HEIGHT or width != self.WIDTH:
            img_2 = cv2.resize(img_2, (self.WIDTH, self.HEIGHT))
        img_2 = (img_2 - self.mean_I) / self.std_I
        img_2 = np.mean(img_2, axis=2, keepdims=True)
        img_2 = np.transpose(img_2, [2, 0, 1])

        org_img = np.concatenate([img_1, img_2], axis=0)

        x = 40
        y = 23
        y_t_flat = np.reshape(self.y_mesh, [-1])
        x_t_flat = np.reshape(self.x_mesh, [-1])
        patch_indices = (y_t_flat + y) * self.WIDTH + (x_t_flat + x)
        input_tesnor = org_img[:, y: y + self.patch_h, x: x + self.patch_w]

        top_left_point = (x, y)
        bottom_left_point = (x, y + self.patch_h)
        bottom_right_point = (self.patch_w + x, self.patch_h + y)
        top_right_point = (x + self.patch_w, y)
        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
        four_points = np.reshape(four_points, (-1))

        org_img = org_img[np.newaxis, :]
        input_tesnor = input_tesnor[np.newaxis, :]
        patch_indices = patch_indices[np.newaxis, :]
        four_points = four_points[np.newaxis, :]
        return (org_img, input_tesnor, patch_indices, four_points)

if __name__=="__main__":
    loader = SingleImgLoader()
    data = loader.getData('../Data/Test/00000238/00000238_10153.jpg', '../Data/Test/00000238/00000238_10156.jpg')
    org_imges = torch.FloatTensor(data[0])
    input_tesnors = torch.FloatTensor(data[1])
    patch_indices = torch.FloatTensor(data[2])
    h4p = torch.FloatTensor(data[3])

    if torch.cuda.is_available():
        org_imges = org_imges.cuda()
        input_tesnors = input_tesnors.cuda()
        patch_indices = patch_indices.cuda()
        h4p = h4p.cuda()


    net = build_model('resnet34', pretrained=False)
    exp_name = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
    model_path = os.path.join(exp_name, 'train_log_Oneline-FastDLT/real_models/resnet34_iter_280000.pth')
    print(model_path)
    state_dict = torch.load(model_path, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.state_dict().items():
        namekey = k[7:]  # remove `module.`
        new_state_dict[namekey] = v
    # load params
    net = build_model('resnet34')
    model_dict = net.state_dict()
    new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict.keys()}
    model_dict.update(new_state_dict)
    net.load_state_dict(model_dict)
    net = torch.nn.DataParallel(net)
    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()

    batch_out = net(org_imges, input_tesnors, h4p, patch_indices)
    H_mat = batch_out['H_mat']
    print (H_mat)