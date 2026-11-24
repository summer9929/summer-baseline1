# DeepLabV3+：odin1 语义分割

本项目以 odin1 为目标，复用原 DeepLabV3+ 模型与分割指标。实现依据为《数据集方案.md》；过程与验证记录见《改进.md》。

## RGB-D 四通道前融合（当前默认）

四通道实现及逐文件修改记录见 [深度四通道改进.md](深度四通道改进.md)。默认输入顺序为 `[R_n,G_n,B_n,D_n]`，不增加模态分支。RGB、Depth、Label 按同一 ID 配对，共同中心裁剪后缩放至高 544、宽 640。深度为 uint16 PNG，按 `raw/256` 解码为米制光轴 Z；不做数值 Clip，有效值用训练集统计量 `mean=2.161799, std=2.206930` 标准化，原始零值在标准化后置零。

`main.py` 原有训练默认值保持不变。以下命令仅增加 RGB 权重迁移入口，不覆盖学习率、batch size 或迭代次数；请在项目根目录执行：

```powershell
& "E:\ANACONDA\envs\vmbev\python.exe" main.py --init_rgb_ckpt "checkpoints/best_deeplabv3plus_mobilenet_odin1_os16.pth"
```

`--init_rgb_ckpt` 仅迁移模型权重，新增深度卷积核置零，重新建立优化器和调度器。不能和 `--ckpt`、`--continue_training` 或 `--test_only` 混用。已有 RGB 权重采用的裁剪和输入尺寸不同，迁移时打印来源配置；本次 RGB-D 始终采用当前 `1088×1280 → 544×640` 配置。

新权重单独保存为 `checkpoints/{best,latest}_deeplabv3plus_mobilenet_odin1_rgbd4_os16.pth`，不会覆盖原 RGB 权重。验证、推理示例：

```powershell
& "E:\ANACONDA\envs\vmbev\python.exe" main.py --test_only --eval_split testing --ckpt "checkpoints/best_deeplabv3plus_mobilenet_odin1_rgbd4_os16.pth" --save_val_results
& "E:\ANACONDA\envs\vmbev\python.exe" predict.py --input "segdataodin/testing/image_data" --ckpt "checkpoints/best_deeplabv3plus_mobilenet_odin1_rgbd4_os16.pth" --save_val_results_to "results/rgbd_predict"
```

推理默认从 `image_data` 同级的 `dense_depth` 找到配对深度。其他路径用 `--depth_input` 指定深度文件或目录。深度文件缺失时报错；文件内无回波像素保留为零。原 RGB 基线可显式使用 `--input_channels 3`，此模式不读取深度；加载原三通道权重时也应指定该参数。下文历史 RGB 权重示例同样需补上 `--input_channels 3`。

## 环境

本次使用用户指定的 `E:\ANACONDA\envs\vmbev\python.exe`：Python 3.11、PyTorch 2.5.1、torchvision 0.20.1。普通训练、评估与推理不要求安装 visdom 或 scikit-learn；只有 `--enable_vis` 才需要 visdom。`calibseg.py` 使用 PyYAML。此次未安装或升级环境依赖。

## 数据结构与类别

```text
segdataodin/
├── calib.yaml
├── training/{image_data,gt_image,dense_depth}/
├── validation/{image_data,gt_image,dense_depth}/
└── testing/{image_data,gt_image,dense_depth}/
```

RGB、深度与灰度标签按文件名配对，当前 training/validation/testing 分别为 18/6/1 组。四通道模式要求 dense_depth 完整；显式三通道 RGB 模式不依赖深度。原始图片、标签、深度和原始标定保持原位。

| 原始灰度值 | 类别 ID | 含义 | RGB |
| ---: | ---: | --- | --- |
| 0 | 0 | 不可通行 | (220, 50, 50) |
| 255 | 1 | 可通行 | (40, 200, 60) |
| 128 | 2 | 天空 | (60, 140, 240) |

先将原始标签编码为 0/1/2，再执行预处理。编码后的填充忽略值为 255，不与原始灰度 255 的可通行语义混用。

## 唯一配置入口与尺寸修改

尺寸与归一化仅在 `main.py` 声明，尺寸变量均按 **(高, 宽)**：

```python
ODIN1_SOURCE_SIZE_HW = (1296, 1600)
ODIN1_CROP_SIZE_HW = (1088, 1280)
ODIN1_INPUT_SIZE_HW = (544, 640)
ODIN1_MEAN = (0.485, 0.456, 0.406)
ODIN1_STD = (0.229, 0.224, 0.225)
```

例如输入改为宽 800、高 672，只修改 `ODIN1_INPUT_SIZE_HW = (672, 800)`。裁剪尺寸不变时，保存结果仍为宽 1280、高 1088。建议输入宽高为 32 的整数倍。

`main.get_transforms()` 显式声明训练和验证流程，`predict.py` 直接复用验证流程。裁剪、缩放、填充、掩码恢复的具体实现位于公共模块；数据集类只读取、编码和调用传入的流程。

```text
1600×1296 原图与编码标签
→ 中心裁剪 1280×1088，裁剪框 (160,104,1440,1192)
→ 缩放 640×544
→ RGB 转张量并归一化，标签保持 int64 类别 ID
```

RGB 双线性缩放，标签最近邻缩放。三个阶段采用相同的确定性流程，无随机裁剪、尺度或翻转。当前尺寸不填充；异常尺寸报出文件和实际大小。公共填充算子保留独立的图像/标签填充值。

## 已训练模型的验证与测试

以下示例权重路径需要替换为实际 odin1 三分类 checkpoint，`--model`、`--output_stride` 和 `--separable_conv` 应与权重一致。测试和推理不会额外下载骨干权重。

```powershell
& "E:\ANACONDA\envs\vmbev\python.exe" main.py --test_only --eval_split validation --ckpt "checkpoints/your_odin1_model.pth" --save_val_results
& "E:\ANACONDA\envs\vmbev\python.exe" main.py --test_only --eval_split testing --ckpt "checkpoints/your_odin1_model.pth" --save_val_results
```

`--eval_split` 默认 validation，仅控制 `--test_only` 的数据划分；正式训练的周期验证始终使用 validation。仅测试模式不构建训练数据集或训练加载器。

指标在网络输入分辨率计算：默认宽 640、高 544。终端打印像素准确率、Mean IoU 和逐类别 IoU，并标明评估尺寸。可视化恢复尺寸不改变指标计算尺度。

默认结果位于 `results/validation/` 或 `results/testing/`，可通过 `--results_dir` 修改。使用原样本 ID 命名，每个评估样本严格保存五类 PNG：

| 后缀 | 内容 | 默认宽×高 |
| --- | --- | --- |
| `_image.png` | 保留的中心裁剪原图 | 1280×1088 |
| `_pred_id.png` | 最近邻恢复的 0/1/2 预测掩码 | 1280×1088 |
| `_overlay.png` | RGB 叠加图 | 1280×1088 |
| `_target_id.png` | 直接保留的裁剪真值类别图 | 1280×1088 |
| `_target_color.png` | 裁剪真值的类别配色图 | 1280×1088 |

`_pred_id.png` 和 `_target_id.png` 使用带调色板的 PNG（P 模式）：底层类别 ID 保持 0/1/2，普通图片查看器中不可通行为黑色 0、可通行为白色 255、天空为灰色 128。使用 PIL 直接转 NumPy 读取时得到类别 ID；需要可见灰度值时使用 `image.convert('L')`。该调色板由数据集原始灰度映射派生，红绿蓝叠加配色不受影响。

不独立保存预测彩色图。预测彩色图只在内存中生成，用于逐像素混合：

```python
overlay = np.clip(np.rint(0.6 * cropped_rgb + 0.4 * class_color), 0, 255).astype(np.uint8)
```

实现中先转 float32 再运算，所有类别包括 0 都参与混合。PIL 直接保存 PNG，保证尺寸与像素内容，不使用 Matplotlib 画布。

## 单图及批量推理

```powershell
& "E:\ANACONDA\envs\vmbev\python.exe" predict.py --input "segdataodin/testing/image_data" --ckpt "checkpoints/your_odin1_model.pth" --save_val_results_to "results/predict2"

#示例
& "E:\ANACONDA\envs\vmbev\python.exe" predict.py --input "segdataodin/testing/image_data" --ckpt "checkpoints/best_deeplabv3plus_mobilenet_odin1_os16.pth" --save_val_results_to "results/predict"

& "E:\ANACONDA\envs\vmbev\python.exe" main.py --test_only --model deeplabv3plus_mobilenet --input "segdataodin/testing/image_data" --ckpt "checkpoints/best_deeplabv3plus_mobilenet_odin1_os16.pth" --save_val_results_to "results/predict3"
```

`--input` 可以为单张原图、RGB 图片目录或含 image_data 的单个划分目录。数据集总目录会提示选择具体划分，gt_image/dense_depth 被拒绝或过滤。批量结果保留相对子目录，同名输出冲突会报错。

无标签推理只保存上述五类中的 `_image.png`、`_pred_id.png`、`_overlay.png`，不伪造真值文件。省略输出目录参数时只执行推理。

## 训练与 checkpoint

```powershell
& "E:\ANACONDA\envs\vmbev\python.exe" main.py --batch_size 4 --val_batch_size 2 --save_val_results

& "E:\ANACONDA\envs\vmbev\python.exe" "E:\Axiangmu\DeepLabV3Plus-Pytorch-master\main.py" --batch_size 2 --val_batch_size 1 --save_val_results

# 原命令现在默认训练 RGB-D 四通道，只训练 RGB 时需要加上 --input_channels 3：
& "E:\ANACONDA\envs\vmbev\python.exe" "E:\Axiangmu\DeepLabV3Plus-Pytorch-master\main.py" --input_channels 3 --batch_size 2 --val_batch_size 1 --save_val_results
#此时只读取 RGB 和标签，不读取深度，网络首层使用三通道，其余训练参数保持原配置。
RGB 模式仍使用原来的 checkpoint 文件名，训练前请备份已有同名 RGB 权重，避免被新训练结果覆盖。

#这个三通道 RGB 权重可以用于初始化四通道 RGB-D 模型，再继续微调，使用新增的 --init_rgb_ckpt：  +训练加入深度后的+

& "E:\ANACONDA\envs\vmbev\python.exe" "E:\Axiangmu\DeepLabV3Plus-Pytorch-master\main.py" --init_rgb_ckpt "E:\Axiangmu\DeepLabV3Plus-Pytorch-master\checkpoints\best_deeplabv3plus_mobilenet_odin1_os16.pth" --batch_size 2 --val_batch_size 1 --save_val_results
#+加载时会：- 复制首层原有 RGB 权重，将新增深度通道权重初始化为零。- 加载其余骨干和分割头权重，联合训练整个网络。- 重新建立优化器、调度器和训练进度。这里使用 --init_rgb_ckpt，不要加 --continue_training。当前默认就是四通道，无需额外指定。


#test only
& "E:\ANACONDA\envs\vmbev\python.exe" "E:\Axiangmu\DeepLabV3Plus-Pytorch-master\main.py" --test_only --eval_split testing --ckpt "E:\Axiangmu\DeepLabV3Plus-Pytorch-master\checkpoints\best_deeplabv3plus_mobilenet_odin1_os16.pth" --save_val_results

#--test_only：只评估，不训练。
测试验证集时，将 --eval_split testing 改为 --eval_split validation。
--save_val_results：保存五类结果到项目的 results/testing 或 results/validation
```

训练使用 SGD 分组学习率、PolyLR 或 StepLR、交叉熵或 FocalLoss。首次训练默认使用骨干预训练参数，可能需要下载；离线随机初始化可传 `--no_pretrained_backbone`。ASPP 含 BatchNorm，训练 batch_size 至少为 2，且不能超过训练样本数。

checkpoint 保存于 `checkpoints/`，包括迭代、优化器、调度器、最佳 mIoU、模型结构、类别语义及预处理配置。`--continue_training --ckpt ...` 恢复训练状态；单独 `--ckpt ...` 加载权重开始新的训练进度。旧 checkpoint 缺少元数据时明确提示无法自动核对的项目，权重形状仍严格检查；已知类别/模型元数据冲突时报错，预处理差异明确打印并使用 main.py 当前配置。

## 派生相机标定

```powershell
& "E:\ANACONDA\envs\vmbev\python.exe" calibseg.py
```

生成 `segdataodin/calib_1280x1088.yaml` 与 `segdataodin/calib_640x544.yaml`，原始 calib.yaml 不覆盖。脚本复用 main.py 的尺寸与公共裁剪记录；修改输入尺寸后重新运行即可获得对应尺寸文件。

裁剪使主点减去 left/top。按整数像素中心与 Pillow 缩放约定，缩放后的主点为 `(u_crop + 0.5) * sx - 0.5`、`(v_crop + 0.5) * sy - 0.5`；A11/A12 乘 sx，A22 乘 sy。畸变参数和 Tcl_0 保持原值；原标定误差统计作为溯源记录保留，不代表新尺寸重新标定的误差。RGB 分割本身不做去畸变，也不读取标定。

后续深度投影应使用与深度/掩码分辨率匹配的派生标定。深度图需另行执行相同裁剪，缩放保留深度单位和无效值语义；本次不修改原始深度图。

## 验证方法

```powershell
& "E:\ANACONDA\envs\vmbev\python.exe" -m unittest discover -s tests -v
```

检查涵盖全量样本、类别编码、几何对齐、三阶段张量一致性、输入尺寸单点修改、填充忽略值、像素混合、五类 PNG、标定投影一致性及原文件保留、真实模型前向、checkpoint 校验与两个入口的保存结果一致性。

测试中的随机模型 checkpoint 仅用于连通性检查，不代表真实精度。实际分割精度需使用匹配的已训练权重，在指定数据划分上评估。

原项目来源：[VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)。网络实现和原许可证保持不变。



### 全部重新训练（1、RGB和2、加入深度的）

可以，按两个阶段依次执行。**第一阶段重新训练 RGB，第二阶段用第一阶段新生成的 RGB 权重初始化 RGB-D。**

下面不加载以前的分割权重；第一阶段默认仍使用 ImageNet 骨干预训练参数。程序会启用 Backbone 的预训练权重。

| 网络组成             | 初始权重来源             |
| -------------------- | ------------------------ |
| MobileNetV2 Backbone | ImageNet 预训练权重      |
| RGB 首层卷积         | MobileNetV2 预训练权重   |
| ASPP                 | 新建模块的初始化权重     |
| Decoder              | 新建模块的初始化权重     |
| 三分类输出层         | 新建模块的初始化权重     |
| Depth 卷积权重       | 不存在，因为输入为三通道 |

具体 ImageNet 权重文件及下载地址需要查看 `network/backbone/mobilenetv2.py` 才能确定。

**1. 重新训练 RGB**

```
& "E:\ANACONDA\envs\vmbev\python.exe" "E:\Axiangmu\DeepLabV3Plus-Pytorch-master\main.py" --input_channels 3 --batch_size 2 --val_batch_size 1 --save_val_results --results_dir "E:\Axiangmu\DeepLabV3Plus-Pytorch-master\results\rgb" --total_itrs 180 --print_interval 9 --val_interval 9 #20轮，每9次迭代打印日志并且用验证集评价。显示指定GPU加速 --gpu_id 0
```

训练结束后，验证集表现最好的 RGB 权重保存为：

```
checkpoints/best_deeplabv3plus_mobilenet_odin1_os16.pth
```

第一阶段会覆盖原有同名权重，需要保留旧模型的话请先备份。

**2. 使用新 RGB 权重训练 RGB-D**

等第一阶段结束后，再执行：

```
& "E:\ANACONDA\envs\vmbev\python.exe" "E:\Axiangmu\DeepLabV3Plus-Pytorch-master\main.py" --input_channels 4 --init_rgb_ckpt "E:\Axiangmu\DeepLabV3Plus-Pytorch-master\checkpoints\best_deeplabv3plus_mobilenet_odin1_os16.pth" --batch_size 2 --val_batch_size 1 --save_val_results --results_dir "E:\Axiangmu\DeepLabV3Plus-Pytorch-master\results\rgbd" --total_itrs 180 --print_interval 9 --val_interval 9
```

第二阶段复制新 RGB 模型的权重，新增深度卷积权重置零，然后联合微调整个网络。最佳权重保存为：

```
checkpoints/best_deeplabv3plus_mobilenet_odin1_rgbd4_os16.pth
```

两阶段都重新建立训练进度，**不要添加 `--continue_training`**。上述命令保留默认的 `lr=0.01`、`total_itrs=30000`。

如果你说的“全部重新训练”还包括 **RGB 骨干也从随机权重开始，不使用 ImageNet 预训练**，在第一阶段命令末尾加：

```
--no_pretrained_backbone
```

第二阶段仍按上面的命令加载第一阶段新训练出的 RGB 权重。
