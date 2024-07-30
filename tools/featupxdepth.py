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

def mem_usage(DEVICE):
    print(f"\nAllocated: {torch.cuda.memory_allocated(DEVICE)/(10**9):.3f} GB(s) Resereved: {torch.cuda.memory_reserved(DEVICE)/(10**9):.3f} GB(s)\n")

class FxD():

    def __init__(self):
        self.model_type= "dinov2"
        self.activation_type= "token"
        self.num_classes=1000
        self.upsampler_type= "jbu_stack"
        
        self.model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        self.encoder = 'vitl'
    
    def load_dino(self, device):
        self.dinov2_model, _, self.dim = get_featurizer(self.model_type, self.activation_type, num_classes=self.num_classes)
        for p in self.dinov2_model.parameters():
            p.requires_grad = False
        self.dinov2_model = self.dinov2_model.to(device)

    def load_featup(self, path, device):
        # path to dinov2_jbu_stack_cocostuff.ckpt
        self.featup_upsampler = get_upsampler(self.upsampler_type, self.dim)
        checkpoint = torch.load(path)
        checkpoint['state_dict'] = OrderedDict([(".".join(k.split('.')[1:]),v) for k,v in checkpoint['state_dict'].items()])
        self.featup_upsampler.load_state_dict(checkpoint['state_dict'])
        self.featup_upsampler = self.featup_upsampler.to(device)

    def load_depth(self, path):
        # path to file depth_anything_vitl14.pth
        self.depth_anything = DepthAnything(self.model_configs[self.encoder])
        self.depth_anything.load_state_dict(torch.load(path))

    def dino(self, image_tensor):
       return self.dinov2_model(image_tensor)
        
    def featupUpsamplper(self, lr_feats, image_tensor):
       return self.featup_upsampler(lr_feats, image_tensor)
    
    def depthanything(self, image_tensor, hw):
        h, w = hw
        depthout = self.depth_anything(image_tensor)
        return self.depth_post_processing(depthout,h,w)
    
    def depth_post_processing(depth_output,h,w):
        depth = F.interpolate(depth_output[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        raw_depth = Image.fromarray(depth.cpu().numpy().astype('uint16'))
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        raw_depth.save(tmp.name)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)
        colored_depth = Image.fromarray(cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1])
        
        return colored_depth
    
    def unit_vec_feats(self, feats):
        magnitude = torch.norm(feats, p=2, dim=1, keepdim=True) + 1e-8 # TO avoid division by 0
        return feats / magnitude

    def fuse(self, depth_feats, rgb_feats):
        return rgb_feats + depth_feats

    def send_all_models_to_device(self, device):
        self.dinov2_model     = self.dinov2_model.to(device)
        self.featup_upsampler = self.featup_upsampler.to(device)
        self.depth_anything   = self.depth_anything.to(device)
    
    def del_dino(self):
        del self.dinov2_model
        torch.cuda.empty_cache()

    def del_featup(self):
        del self.featup_upsampler
        torch.cuda.empty_cache()

    def del_depth(self):
        del self.depth_anything
        torch.cuda.empty_cache()
