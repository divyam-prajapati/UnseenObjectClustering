from featup.featurizers.util import get_featurizer
from featup.upsamplers import get_upsampler
import torch
from collections import OrderedDict
from PIL import Image
import torchvision.transforms as T
from featup.util import norm, unnorm
import cv2
from torchvision.io import read_image
import numpy as np
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import tempfile
import torch.nn.functional as F
from featup.plotting import plot_feats

def networkFxD():
    network = FxD()
    return network

class FxD():
    def __init__(self):
        model_type= "dinov2"
        activation_type= "token"
        num_classes=1000
        self.dinov2_model, _, dim = get_featurizer(model_type, activation_type, num_classes=num_classes)
        for p in self.dinov2_model.parameters():
            p.requires_grad = False
        upsampler_type= "jbu_stack"
        self.featup_upsampler = get_upsampler(upsampler_type, dim)
        checkpoint = torch.load("D:/CODE/Unseen-Object-Segmentation/checkpoints/dinov2_jbu_stack_cocostuff.ckpt")
        checkpoint['state_dict'] = OrderedDict([(".".join(k.split('.')[1:]),v) for k,v in checkpoint['state_dict'].items()])
        self.featup_upsampler.load_state_dict(checkpoint['state_dict'])
        # model_configs = {
        # 'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        # }
        # encoder = 'vitl'
        # self.depth_anything = DepthAnything(model_configs[encoder])
        # self.depth_anything.load_state_dict(torch.load(f'D:/CODE/Unseen-Object-Segmentation/checkpoints/depth_anything_vitl14.pth'))
   
    def send_model_to_device(self, device):
        self.dinov2_model     = self.dinov2_model.to(device)
        self.featup_upsampler = self.featup_upsampler.to(device)
        # self.depth_anything   = self.depth_anything.to(device)

    def dino(self, image_tensor):
       return self.dinov2_model(image_tensor)
    
    def del_dino(self):
        del self.dinov2_model
        torch.cuda.empty_cache()
    
    def featupUpsamplper(self, lr_feats, image_tensor):
       return self.featup_upsampler(lr_feats, image_tensor)
    
    def depthanything(self, image_tensor, hw):
        h, w = hw
        depthout = self.depth_anything(image_tensor)
        depth = F.interpolate(depthout[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        raw_depth = Image.fromarray(depth.cpu().numpy().astype('uint16'))
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        raw_depth.save(tmp.name)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)
        colored_depth = Image.fromarray(cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1])
        
        return colored_depth
