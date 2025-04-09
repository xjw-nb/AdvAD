import subprocess
from pathlib import Path
import sys
import os

# 获取项目根目录：假设 run_all_inv3.py 在 Inv3 目录下
project_root = Path(__file__).resolve().parent.parent
os.environ["PYTHONPATH"] = str(project_root)
#  文件夹路径
res50_dir = project_root / "Res50"

# 当前 Python 解释器路径（确保是虚拟环境里的）
python_exec = sys.executable

# 要运行的脚本列表（都在 Inv3 下）
scripts = [
    "b45.py",
    "b410.py",
    "b420.py",
    "b85.py",
    "b810.py",
    "b820.py",
    "b165.py",
    "b1610.py",
    "b1620.py",
    "b325.py",
    "b3210.py",
    "b3220.py"
]

for script_name in scripts:
    script_path = res50_dir / script_name
    print(f"正在运行：{script_path}")

    result = subprocess.run(
        [python_exec, str(script_path)],
        cwd=project_root  # 工作目录始终设为项目根目录
    )

    if result.returncode != 0:
        print(f"脚本 {script_name} 运行失败，退出码：{result.returncode}")
        break
    else:
        print(f"{script_name} 运行成功\n")
