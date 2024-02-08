import argparse
from mmengine.runner import CheckpointLoader, save_checkpoint
from mmpretrain.models import VisionTransformer
import torch
# dinov2 = VisionTransformer('large')
# print(dinov2)

def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained segformer to '
        'MMSegmentation style.')
    parser.add_argument('--src', default='/Users/huangjianxin/Documents/learning/Depth-Anything/checkpoints/depth_anything_vitl14.pth', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    # parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    ori_dict = list(state_dict.keys())
    ori_dict = [f for f in ori_dict if 'pretrain' in f]
    # print(ori_dict)
    new_keys = []
    new_static_dict = {}
    for k in ori_dict:
        new_k = k.replace('pretrained.', '').replace('blocks', 'layers')
        if 'ls1' in new_k:
            new_k = new_k.replace('ls1', 'attn') + '1.weight'
            # print(new_k)
            # print(state_dict[k])
        if 'ls2' in new_k:
            new_k = new_k.replace('ls2', 'ffn') + '2.weight'
        if 'norm1' in new_k:
            new_k = new_k.replace('norm1', 'ln1')
        if 'norm2' in new_k:
            new_k = new_k.replace('norm2', 'ln2')
        if 'mlp' in new_k:
            new_k = new_k.replace('mlp', 'ffn')
        if 'fc1' in new_k:
            new_k = new_k.replace('fc1', 'layers.0')
        if 'fc2' in new_k:
            new_k = new_k.replace('fc2', 'layers.1')
        if 'ffn.layers.0' in new_k:
            new_k = new_k.replace('ffn.layers.0', 'ffn.layers.0.0')
        if 'patch_embed.proj' in new_k:
            new_k = new_k.replace('proj', 'projection')
        if  'norm.weight' in k or 'norm.bias' in k:
            new_k = new_k.replace('norm', 'ln1')
        if 'mask_token' in k:
            continue
        new_keys.append(new_k)
        new_static_dict[new_k] = state_dict[k]

    # save_checkpoint(new_static_dict, "/Users/huangjianxin/Documents/learning/Depth-Anything/checkpoints/mmseg_depthenything_vitl14.pth")
    # new_keys.sort()
    print(new_keys)
    print()
    mmdinov2 = VisionTransformer('large', layer_scale_init_value=0.1)
    model_dict = list(mmdinov2.state_dict().keys())
    # model_dict.sort()
    print(model_dict)
    print()
    print(list(set(new_keys)-set(model_dict)))

# main()
mmdinov2_1 = VisionTransformer('large',
                               img_size=512,
                               patch_size=14,
                               layer_scale_init_value=0.1,
                               init_cfg=dict(type='Pretrained', checkpoint='/Users/huangjianxin/Documents/learning/Depth-Anything/checkpoints/mmseg_depthenything_vitl14.pth'))
# # print()
new_model_static = torch.load('/Users/huangjianxin/Documents/learning/Depth-Anything/checkpoints/mmseg_depthenything_vitl14.pth')
mmdinov2_1.load_state_dict(new_model_static)
new_model_static = mmdinov2_1.state_dict()
print(new_model_static['layers.0.attn.gamma1.weight'])


model_config = dict(
    type='DINOv2VisionTransformerPretrained',
    arch='large',
    img_size=512,
    patch_size=14,
    layer_scale_init_value=0.1,
    init_cfg=dict(type='Pretrained', checkpoint='/Users/huangjianxin/Documents/learning/Depth-Anything/checkpoints/mmseg_depthenything_vitl14.pth')
)
# model_config.pretrained = '/Users/huangjianxin/Documents/learning/Depth-Anything/checkpoints/mmseg_depthenything_vitl14.pth'



# from mmengine.registry import MODELS
import sys
sys.path.append('/Users/huangjianxin/Documents/learning/mmsegmentation')
from mmseg.registry import MODELS

model = MODELS.build(model_config)
# model.init_weights()
new_model_static = model.state_dict()
print(new_model_static['layers.0.attn.gamma1.weight'])
