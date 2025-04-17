import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
from art.estimators.classification import PyTorchClassifier
import timm
from timm import create_model
from Normalize import Normalize, TfNormalize
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    )
import warnings
import pytorch_fid.fid_score as fid_score

warnings.filterwarnings("ignore")

def get_vim_model(type="small"):
    import sys

    # sys.path.append('/Dir to the VisionMamba code of/Vim-main/vim')
    sys.path.append('Vim-main/Vim-main/vim')

    import models_mamba
    if type == "small":
        model_name = 'vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2'
    elif type == "tiny":
        model_name = 'vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2'
    else:
        assert False
    print("Creating Vim")
    model = create_model(
        model_name=model_name,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=224
    )

    # if type == "small":
    #     path = "Dir to the VisionMamba code of Vim-main/vim/ckpts/vim_s_midclstok_80p5acc.pth"
    # elif type == "tiny":
    #     path = "Dir to the VisionMamba code of Vim-main/vim/ckpts/vim_t_midclstok_76p1acc.pth"
    # else:
    #     assert False
    if type == "small":
        path = "checkpoints/vim_s_midclstok_80p5acc.pth"
    elif type == "tiny":
        path = "checkpoints/vim_t_midclstok_76p1acc.pth"
    else:
        assert False
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    return model

# def get_model(name):
#     model_dir = "/home/ubuntu/.cache/torch/hub/checkpoints/"
#     model_path = os.path.join(model_dir, name + '.npy')
#     if name == 'tf2torch_inception_v4':
#         model = tf_inception_v4
#     elif name == 'tf2torch_inc_res_v2':
#         model = tf_inc_res_v2
#     elif name == 'tf2torch_resnet_v2_101':
#         model = tf_resnet_v2_101
#     elif name == 'tf2torch_resnet_v2_101':
#         model = timm.create_model('vit_small_patch16_224', pretrained=True,num_classes=1000, global_pool='')
#     else:
#         raise NotImplementedError("No such model!")
#     model = nn.Sequential(
#         # Images for inception classifier are normalized to be in [-1, 1] interval.
#         TfNormalize('tensorflow'),
#         model.KitModel(model_path).eval().cuda(), )
#     return model.cuda()
def model_selection(name):
    # model_dir = "/home/ubuntu/.cache/torch/hub/checkpoints/"
    # model_path = os.path.join(model_dir, name + '.npy')
    if name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif name == "convnext":
        model = models.convnext_base(pretrained=True)
    elif name == "swin":
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    elif name == "vim_s_midclstok_80p5acc":
        model = get_vim_model("small")
    elif name == "vim-tiny":
        model = get_vim_model("tiny")
    elif name == "vgg19":
        model = models.vgg19(pretrained=True)
    elif name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif name == "mobile_v2":
        model = models.mobilenet_v2(pretrained=True)
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=True)
    elif name == "wrn50":
        model = models.wide_resnet50_2(pretrained=True)
    elif name == "resnet101":
        model = models.resnet101(pretrained=True)
    elif name == "inception_v4":
        model = timm.create_model('inception_v4.tf_in1k', pretrained=False)
        state_dict = torch.load('/home/ubuntu/models/inception_v4.pth')
        model.load_state_dict(state_dict)
    elif name == "inception_resnet_v2":
        model = timm.create_model('inception_resnet_v2.tf_in1k', pretrained=False)
        state_dict = torch.load('/home/ubuntu/models/inception_resnet_v2.pth')
        model.load_state_dict(state_dict)
    elif name == "mobilenetv3":
        model = timm.create_model('mobilenetv3_small_100.lamb_in1k', pretrained=False)
        state_dict = torch.load('/home/ubuntu/models/mobilenetv3.pth')
        model.load_state_dict(state_dict)
    elif name == "adv_inception_v3":
        model = timm.create_model('inception_v3.tf_adv_in1k', pretrained=False)
        state_dict = torch.load('/home/ubuntu/models/adv_inception_v3.pth')
        model.load_state_dict(state_dict)

    # elif name == 'tf2torch_inception_v4':
    #     model = tf_inception_v4
    # elif name == 'tf2torch_resnet_v2_50':
    #     model = tf_resnet_v2_50
    # elif name == 'tf2torch_resnet_v2_101':
    #     model = tf_resnet_v2_101
    # elif name == "tf2torch_inc_res_v2":
    #     model = timm.create_model('inception_resnet_v2', pretrained=True)
    # elif name == "resnet_v2_101":
    #     model = timm.create_model('resnetv2_101x1_bitm', pretrained=False)
    else:
        raise NotImplementedError("No such model!")
    # model = nn.Sequential(
    #     # Images for inception classifier are normalized to be in [-1, 1] interval.
    #     TfNormalize('tensorflow'),
    #     model.KitModel(model_path).eval().cuda(),)
    return model.cuda()

if __name__ == "__main__":
    model = model_selection("vim-tiny")
    print(model)