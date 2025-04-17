"""
The code about diffusion initialization is based on the open-source code of paper:
"Diffusion Models Beat GANs on Image Synthesis"
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np

import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime
import time
from natsort import index_natsorted

from torch.utils.data import DataLoader
from torchcam.methods import GradCAM
from torchvision import transforms

from ImageNet_Compatible_Dataset import ImageNet_Compatible
from eval_all import eval_all
from utils import logger
import attacked_models
from utils.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict, create_diffusion
)
from PIL import Image

from pytorch_grad_cam import GradCAM as GradCAM_PY
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

##load image metadata (Image_ID, true label, and target label)
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_ori_list.append(int(row['TrueLabel']) - 1)
            label_tar_list.append(int(row['TargetClass']) - 1)

    sorted_indices = index_natsorted(image_id_list)
    image_id_list = [image_id_list[i] for i in sorted_indices]
    label_ori_list = [label_ori_list[i] for i in sorted_indices]
    label_tar_list = [label_tar_list[i] for i in sorted_indices]

    return image_id_list, label_ori_list, label_tar_list


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

# prev_grad = None
def attack_main_advadx(model_name=None):
    device = th.device("cuda")
    args = create_attack_argparser().parse_args()

    if model_name is not None:
        args.model_name = model_name
    else:
        model_name = args.model_name

    if (args.model_name == "swin" or args.model_name == "convnext") and args.batch_size > 50:
        args.batch_size = 50

    if args.manual_seed is not None:
        print(args.manual_seed)
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        th.manual_seed(args.manual_seed)
        th.cuda.manual_seed(args.manual_seed)
        th.cuda.manual_seed_all(args.manual_seed)
        th.benchmark = False
        th.deterministic = True
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
        # th.use_deterministic_algorithms(True)

    save_dir = 'attack_results/'
    today = datetime.date.today()
    now = time.strftime("_%H%M%S")
    file_name = "attack" + (str(today).replace('-', '') + now)
    save_dir = os.path.join(save_dir, file_name)
    os.makedirs(save_dir, exist_ok=True)

    logger.configure(dir=save_dir)
    logger.log(args)
    logger.log("creating model and diffusion...")
    temp_dict = model_and_diffusion_defaults()
    temp_dict.update({"attack_method": args.attack_method})
    temp_dict.update({"image_size": args.image_size})

    args_dict = vars(args)
    args_dict.update(temp_dict)
    args = argparse.Namespace(**args_dict)

    # creating non-parametric diffusion process
    attack_diffusion = create_diffusion(
        **args_to_dict(args, temp_dict.keys())
    )
    eval_model = args.eval_model
    model_name = args.model_name

    attacked_models_list = [
        attacked_models.model_selection("vgg19").eval(),  # 模型1
        attacked_models.model_selection("mobile_v2").eval(),  # 模型2
        attacked_models.model_selection("inception_v3").eval(),  # 模型3
    ]
    attacked_model = attacked_models.model_selection(model_name).eval()

    def AMG_grad_func_DGI(x, t, y, eps, attack_type=None):
        assert attack_type is not None
        # global prev_grad
        momentum_factor = 0.3

        timestep_map = attack_diffusion.timestep_map
        rescale_timesteps = attack_diffusion.rescale_timesteps
        original_num_steps = attack_diffusion.original_num_steps
        map_tensor = th.tensor(timestep_map, device=t.device, dtype=t.dtype)
        new_ts = map_tensor[t]
        if rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / original_num_steps)

        with th.enable_grad():
            xt = x.detach().clone().requires_grad_(True)
            pred_xstart = attack_diffusion._predict_xstart_from_eps(xt, t, eps)  # 基于xt估计x0
            pred_xstart = (pred_xstart / 2 + 0.5).clamp(0, 1)
            pred_xstart = pred_xstart.permute(0, 2, 3, 1)

            if "adv" in args.model_name:
                # preprocess_np = (np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
                mean = th.as_tensor([0.5, 0.5, 0.5], dtype=pred_xstart.dtype, device=pred_xstart.device)
                std = th.as_tensor([0.5, 0.5, 0.5], dtype=pred_xstart.dtype, device=pred_xstart.device)
            elif "Singh" in args.model_name:
                # preprocess_np = (np.array([0, 0, 0]), np.array([1, 1, 1]))
                mean = th.as_tensor([0, 0, 0], dtype=pred_xstart.dtype, device=pred_xstart.device)
                std = th.as_tensor([1, 1, 1], dtype=pred_xstart.dtype, device=pred_xstart.device)
            else:
                mean = th.as_tensor([0.485, 0.456, 0.406], dtype=pred_xstart.dtype, device=pred_xstart.device)
                std = th.as_tensor([0.229, 0.224, 0.225], dtype=pred_xstart.dtype, device=pred_xstart.device)

            pred_xstart = pred_xstart[:, :, :].sub(mean).div(std)
            pred_xstart = pred_xstart.permute(0, 3, 1, 2)
            x_start = pred_xstart.clone().requires_grad_(True) #detacth
            logits = attacked_model(pred_xstart)  # 计算估计的x0的logits
            choice = th.argmax(logits, 1).detach() == y  # 寻找logits最大的索引对应的类别，再与原标签进行比较获得布尔张量choice
            grad_zeros = th.zeros_like(x)  # 创建一个全是0的张量

            if not choice.any():
                return grad_zeros, choice
            grads = []
            for model in attacked_models_list:  # 尝试每100次算一次梯度
                logits = model(x_start)  # 当前模型的 logits
                if attack_type == "target":
                    log_probs = F.log_softmax(logits, dim=-1)
                    log_probs_selected = log_probs[range(len(logits)), y.view(-1)]
                    grad = th.autograd.grad(log_probs_selected.sum(), xt)[0]
                elif attack_type == "untarget":
                    probs = F.softmax(logits, dim=-1)
                    probs_selected = probs[range(len(logits)), y.view(-1)]
                    zero_nums = (probs_selected == 1) * 1e-6
                    log_one_minus_probs_selected = th.log(1-probs_selected + zero_nums)
                    grad = th.autograd.grad(log_one_minus_probs_selected.sum(), xt, retain_graph=True)[0]
                else:
                    assert False
                grads.append(grad)
                # 合并梯度 (例如取平均)
            final_grad = sum(grads) / len(grads)
            m_svrg = 1  # 改为3
            momentum = 1.0
            # x_t1 = xt.detach().clone().requires_grad_(True)
            x_m = x_start.clone().requires_grad_(True)  # 外层循环估计的x_0
            G_m = th.zeros_like(xt, device=xt.device)
            # 迭代进行梯度更新
            for j in range(m_svrg):
                beta = 2. / 2550
                # 使用同一模型来计算 x_m 的梯度
                model_index = th.randint(len(attacked_models_list), (1,)).item()  # 随机选择一个模型索引
                model = attacked_models_list[model_index]
                logits = model(x_m)  # 计算模型的 logits
                if attack_type == "target":
                    log_probs = F.log_softmax(logits, dim=-1)
                    log_probs_selected = log_probs[range(len(logits)), y.view(-1)]
                    grad_xj = th.autograd.grad(log_probs_selected.sum(), x_m)[0]
                elif attack_type == "untarget":
                    probs = F.softmax(logits, dim=-1)
                    probs_selected = probs[range(len(logits)), y.view(-1)]
                    zero_nums = (probs_selected == 1) * 1e-6
                    log_one_minus_probs_selected = th.log(1-probs_selected + zero_nums)
                    grad_xj = th.autograd.grad(log_one_minus_probs_selected.sum(), x_m)[0]
                else:
                    assert False
                # 随机选择一个模型的梯度
                grad_single = grads[model_index]  # 从梯度列表中选择与模型匹配的梯度
                # 计算梯度差异
                g_m = grad_xj - (grad_single - final_grad)
                # print("1111",grad_single-grad_xj)
                # 使用 momentum 更新梯度
                G_m1 = momentum * G_m + g_m
                x_j = x_m + beta * th.sign(G_m1)  # 通过符号函数更新 x_j
                x_j = th.clamp(x_j, 0.0, 1.0)
                x_m = x_j
                # 更新 G_m
                G_m = G_m1  # 记得更新 G_m，用于下一次迭代

            momentum_grad = momentum_factor * final_grad + (1 - momentum_factor) * G_m
            # # 更新 prev_grad 为当前梯度的副本，供下次迭代使用
            prev_grad = momentum_grad.detach().clone()  # 创建当前梯度的副本

        return prev_grad, choice

    images_root = "/home/ubuntu/AdvAD/AdvAD-main/dataset1/images"  # The clean images' root directory.
    image_id_list, label_ori_list, label_tar_list = load_ground_truth('/home/ubuntu/AdvAD/AdvAD-main/dataset1/images.csv')

    assert len(image_id_list) == len(label_ori_list) == len(label_tar_list)

    '''GradCAM Assistance'''
    if "vim" in model_name:
        pass
    elif model_name != "swin" and model_name != "Liu_SwinB":
        if model_name == "resnet50" or model_name == "wrn50" or model_name == "SalmanR50":
            target_layer = "layer4"
        elif model_name == "inception_v3":
            target_layer = "Mixed_7c"
        elif model_name == "vgg19":
            target_layer = "features.36"
        elif model_name == "mobile_v2":
            target_layer = "features.17"
        elif model_name == "convnext":
            target_layer = "features"
        elif model_name == "Liu_ConvNextB" or model_name == "Singh_ConvNextB":
            target_layer = "stages"
        elif model_name == "new_tf2torch_adv_inception_v3":
            target_layer = "ConcatLayer"
        elif model_name == "new_tf2torch_ens3_adv_inc_v3":
            target_layer = "ConcatLayer"
        elif model_name == "new_tf2torch_ens4_adv_inc_v3":
            target_layer = "ConcatLayer"
        elif model_name == "new_tf2torch_ens_adv_inc_res_v2":
            target_layer = "ReLULayer"
        else:
            assert False
        cam_extractor = GradCAM(attacked_model, target_layer)
    else:
        def reshape_transform(tensor, height=7, width=7):
            result = tensor.permute(0, 3, 1, 2)
            return result
        target_layers = [attacked_model.features] if model_name == "swin" else [attacked_model.layers]
        cam_extractor = GradCAM_PY(model=attacked_model, target_layers=target_layers,
                                   reshape_transform=reshape_transform)


    all_adv_images = []
    all_BP_iter_count = 0

    batchsize = args.batch_size
    image_dataset = ImageNet_Compatible(root="/home/ubuntu/AdvAD/AdvAD-main/dataset1", image_size=args.image_size)
    data_loader = DataLoader(image_dataset, batch_size=batchsize, shuffle=False, drop_last=False)

    start_time = time.time()
    for batch_i, (inputs_all, label_ori_all, label_tar_all) in enumerate(data_loader):
        grads = []
        prev_grad = None
        print("samples: {} / {}".format((batch_i + 1) * batchsize, len(image_dataset)))
        x0_ori = inputs_all.to(device)
        prev_grad = th.zeros_like(x0_ori, device=x0_ori.device)
        label_ori_all = label_ori_all.to(device)
        label_tar_all = label_tar_all.to(device)

        """
        attack start
        """
        model_kwargs = {}

        model_kwargs["y_ori"] = label_ori_all
        model_kwargs["y_tar"] = label_tar_all

        '''GradCAM Assistance Mask'''
        if "adv" in args.model_name:
            transform = transforms.Compose([transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5])
            ])
            input = transform(x0_ori)
        elif "Singh" in args.model_name:
            input = x0_ori
        else:
            transform = transforms.Compose([transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            ])
            input = transform(x0_ori)

        mask_new = []
        for i in range(input.shape[0]):

            if "vim" in model_name:
                pass
            elif model_name != "swin" and model_name != "Liu_SwinB":
                scores = attacked_model(input[i].unsqueeze(0))  # 预测分数
                if args.attack_type == "untarget":
                    mask_now1 = cam_extractor(label_ori_all[i].item(), scores)[0][0]
                elif args.attack_type == "target":
                    mask_now1 = cam_extractor(label_ori_all[i].item(), scores)[0][0]
                else:
                    assert False


                mask_now1 = mask_now1.detach().cpu().numpy()
                mask_now1 = cv2.resize(mask_now1, (args.image_size, args.image_size), interpolation=cv2.INTER_CUBIC)

                # 计算块大小
                block_size = 16
                num_blocks = (mask_now1.shape[0] // block_size, mask_now1.shape[1] // block_size)

                # 存储每个块的特征重要性
                block_importance = np.zeros(num_blocks)

                # 遍历每个块并计算特征重要性
                for i in range(num_blocks[0]):
                    for j in range(num_blocks[1]):
                        block = mask_now1[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                        # 计算每个块的特征重要性，比如使用均值作为特征重要性
                        # block_importance[i, j] = np.mean(block)  # 算术平均值
                        # block_importance[i, j] = np.sqrt(np.mean(np.square(block)))  # 均方根均值
                        block_importance[i, j] = np.linalg.norm(block, ord=2)  # l2范数

                # 计算前50%的块
                threshold = np.percentile(block_importance, 50)
                important_blocks = np.where(block_importance >= threshold)

                # 创建一个新掩码，初始化为零
                new_mask = np.zeros_like(mask_now1)

                # 保留前50%块中的20%像素
                for block_i, block_j in zip(*important_blocks):
                    block = mask_now1[block_i * block_size:(block_i + 1) * block_size,
                            block_j * block_size:(block_j + 1) * block_size]

                    # 在块内选出前20%的像素
                    pixel_threshold = np.percentile(block, 10)
                    new_mask[block_i * block_size:(block_i + 1) * block_size,
                    block_j * block_size:(block_j + 1) * block_size] = np.where(block >= pixel_threshold, block, 0)

                mask_new.append(new_mask)
            else:
                targets = [ClassifierOutputTarget(label_ori_all[i].item())]
                mask_now = cam_extractor(input_tensor=input[i].unsqueeze(0), targets=targets)[0]  # 这里获得每个输入的掩码
                mask_now = mask_now.astype(np.uint8)
                mask_now = Image.fromarray(mask_now)
                #mask_all.append(mask_now)
        if "vim" in model_name:
            mask_ori = None
        else:
            #mask_ori = th.from_numpy(np.stack(mask_all, axis=0)).to(device).unsqueeze(1).clamp(0, 1)  # 把所有图像的掩码进行拼接堆叠然后转化为张量
            mask_ori1 = th.from_numpy(np.stack(mask_new, axis=0)).to(device).unsqueeze(1).clamp(0, 1)

        x0_ori = 2.0 * x0_ori - 1.0  # [-1, 1]
        eps_ori = th.randn((batchsize, 3, args.image_size, args.image_size)).cuda()  # 随机初始扰动

        iter_num = attack_diffusion.num_timesteps

        T = th.tensor([iter_num - 1], dtype=th.int64, device=device)
        x_T_ori = (_extract_into_tensor(attack_diffusion.sqrt_alphas_cumprod, T, x0_ori.shape) * x0_ori  # 获得T步加噪的图像
                   + _extract_into_tensor(attack_diffusion.sqrt_one_minus_alphas_cumprod, T,
                                          x0_ori.shape) * eps_ori)

        '''Start Attacks in Diffusing'''
        results, BP_iter_count = attack_diffusion.adversarial_attacks_in_diffusing(
            (args.batch_size, 3, args.image_size, args.image_size),
            budget_Xi=args.budget_Xi,
            mask_ori1=mask_ori1,
            x0_ori=x0_ori,
            xt_ori_list=None,
            eps_ori_list=[eps_ori],
            noise=x_T_ori,
            model_kwargs=model_kwargs,
            attack_type=args.attack_type,
            AMG_grad_func_DGI=AMG_grad_func_DGI,
            device=device,
        )

        all_BP_iter_count += BP_iter_count.sum().item()
        # logger.log("BP iter count: {}".format(BP_iter_count))

        """
        attack end
        """
        # sample_output = results[-1]["proj_sample"][0]
        sample_output = results["proj_sample"]
        sample_output = (sample_output / 2. + 0.5).clamp(0, 1)
        adv_image = (sample_output * 255).clamp(0, 255)
        x0 = ((x0_ori + 1.0) / 2.0 * 255).clamp(0, 255)
        noise_image = adv_image - x0
        # 保存对抗样本
        for i in range(adv_image.shape[0]):
            adv_image_test = adv_image[i]
            adv_image_test = adv_image_test.byte().cpu().numpy().transpose(1, 2, 0)

            # 使用 PIL 保存图像
            image_test = Image.fromarray(adv_image_test)
            image_test.save(os.path.join(save_dir, f'adversarial_example_batch{batch_i}_sample{i}.png'))

            noise_image_test = noise_image[i].byte().cpu().numpy().transpose(1, 2, 0)
            noise_image_pil = Image.fromarray(noise_image_test)
            noise_image_pil.save(os.path.join(save_dir, f'noise_example_batch{batch_i}_sample{i}.png'))
        all_adv_images.append(adv_image)

        # if (batch_i + 1) * args.batch_size >= 20:
        #     break

    end_time = time.time()
    logger.log("Run time: {}".format(end_time - start_time))

    # all_adv_images = np.stack(all_adv_images, axis=0)
    all_adv_images = th.cat(all_adv_images, dim=0).detach().cpu().numpy()
    all_adv_images = all_adv_images.astype(np.float32) / 255.0

    out_path = os.path.join(save_dir, "adversarial_examples.npz")
    logger.log("Saving to {}".format(out_path))
    np.savez(out_path, all_adv_images)
    th.cuda.empty_cache()

    img_num = len(all_adv_images)
    logger.log("Avg BP Iter: {}".format(all_BP_iter_count / img_num))
    logger.log("****************** Evaluation start ******************")
    eval_all(file_name=file_name, re_logger=False, quant=False, eval_model=eval_model)
    logger.log("****************** Evaluation complete ******************")

def create_attack_argparser():
    defaults = dict(
        timestep_respacing="ddim1000",  # control step T with 'ddimT'
        noise_schedule="linear",
        classifier_scale=1,
        use_ddim=True,
        manual_seed=123,
        eta=0,

        batch_size=50,
        budget_Xi=8,  # 0-255, PC
        attack_method="AdvAD-X",
        attack_type="untarget",
        image_size=224,

        model_name="mobile_v2",
        eval_model="resnet50",
        model_name_list=["vgg19", "mobile_v2", "inception_v3"]
        # model_name="inception_v3",
        # model_name="swin",
        # model_name="mobile_v2",
        # model_name="vgg19",
        # model_name="wrn50",
        # model_name="convnext",
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    attack_main_advadx()

