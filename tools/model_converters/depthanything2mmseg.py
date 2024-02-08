import argparse
import os.path as osp
import mmengine
from mmengine.runner import CheckpointLoader, save_checkpoint

import torch

def convert_dinov2(state_dict):
    ori_dict = list(state_dict.keys())
    ori_dict = [f for f in ori_dict if 'pretrain' in f]
    new_keys = []
    new_static_dict = {}
    for k in ori_dict:
        new_k = k.replace('pretrained.', '').replace('blocks', 'layers')
        if 'ls1' in new_k:
            new_k = new_k.replace('ls1', 'attn') + '1.weight'
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
    return new_static_dict

def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained depthanything dinov2 to '
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    new_state_dict = convert_dinov2(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(new_state_dict, args.dst)

if __name__ == '__main__':
    main()