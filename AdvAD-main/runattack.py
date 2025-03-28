import argparse
import os
from sparse_run import attack_main_advadx

def run_attack_with_momentum():
    # 设置基础参数
    momentum_factors = [i * 0.1 for i in range(1, 11)]  # 从 0.1 到 1.0 依次增加

    for momentum_factor in momentum_factors:
        print(f"Running attack momentum factor {momentum_factor}")

        # 设置动量因子
        args = argparse.Namespace(momentum_factor=momentum_factor, model_name=None)
        # 你可以将 args 传递给攻击函数，也可以根据需要调整攻击代码
        attack_main_advadx(momentum_factor=args.momentum_factor,model_name=args.model_name)  # 调用主攻击函数


if __name__ == "__main__":
    run_attack_with_momentum()
