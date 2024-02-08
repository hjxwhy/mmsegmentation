import sys
sys.path.append('/Users/huangjianxin/Documents/learning/mmsegmentation')

import torch
from mmseg.models import DINOv2VisionTransformerPretrained

model_config = dict(
    arch='large',
    img_size=518,
    patch_size=14,
    layer_scale_init_value=0.1,
    reshape_to_patched_size=True,
    init_cfg=dict(type='Pretrained', checkpoint='/Users/huangjianxin/Documents/learning/Depth-Anything/checkpoints/mmseg_depthenything_vitl14.pth')
)

dinov2 = DINOv2VisionTransformerPretrained(**model_config)
dinov2.eval()

imgs = torch.randn(1, 3, 518, 518)

with torch.no_grad():
    outputs = dinov2(imgs)
print(outputs[0])
    
class DINOv2(torch.nn.Module):
    """Use DINOv2 pre-trained models
    """

    def __init__(self, version='large', freeze=True, load_from="/Users/huangjianxin/Documents/learning/Depth-Anything/checkpoints/depth_anything_vitl14.pth"):
        super().__init__()
        
        if version == 'large':
            self.dinov2 = torch.hub.load('/Users/huangjianxin/Documents/learning/Depth-Anything/torchhub/facebookresearch_dinov2_main', 'dinov2_vitl14', source='local', pretrained=False)
        else:
            raise NotImplementedError

        if load_from is not None:
            d = torch.load(load_from, map_location='cpu')
            new_d = {}
            for key, value in d.items():
                if 'pretrained' in key:
                    new_d[key.replace('pretrained.', '')] = value
            self.dinov2.load_state_dict(new_d)
        
        self.freeze = freeze
        
    def forward(self, inputs):
        B, _, h, w = inputs.shape
        
        if self.freeze:
            with torch.no_grad():
                features = self.dinov2.get_intermediate_layers(inputs, 4)
        else:
            features = self.dinov2.get_intermediate_layers(inputs, 4)
        
        outs = []
        for feature in features:
            C = feature.shape[-1]
            feature = feature.permute(0, 2, 1).reshape(B, C, h // 14, w // 14).contiguous()
            outs.append(feature)
        
        return outs

ori_dinov2 = DINOv2()
ori_outputs = ori_dinov2(imgs)
print(ori_outputs[0])
print(outputs[1] - ori_outputs[1])