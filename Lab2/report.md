# 实验二：傅里叶变换实验报告

本报告总结了实验二的任务和研究结果，重点探讨了傅里叶变换的特性及其在信号处理和复数神经网络中的应用。实验是在 Marimo 笔记本环境中进行的。

---

## 实践一：傅里叶变换的频谱特性

### 任务 1.1：选择喜欢的音乐或图片，完成以下任务

本任务旨在通过对音频或图像信号进行傅里叶变换，观察其振幅谱和相位谱的特性，并设计实验探究在信号恢复过程中振幅谱和相位谱的重要性，同时在傅里叶频域实现特定频段的滤波效果。

#### 傅里叶变换的实现与观察

在 Python 平台上，我们调用现有的库函数实现了傅里叶变换，并生成了振幅谱和相位谱。具体实现中，代码提供了`fft_1d`（针对音频）和`fft_2d`（针对图像）函数，利用用户选择的库（SciPy、NumPy 或 PyTorch，通过`select_fft_ifft_module`函数选择）中的 FFT 功能（如`fft`、`fft2`、`fftshift`）进行变换。

-   对于一维音频信号，代码加载音频文件，执行 FFT，并生成图表（`./outputs/audio_fft_analysis.png`），展示了原始波形、振幅谱和相位谱。
-   对于二维图像信号，代码加载图像（必要时转换为灰度图），执行二维 FFT，并生成图表（`./outputs/image_fft_analysis.png`），展示了原始图像、对数振幅谱（经过平移）和相位谱（经过平移）。

通过这些可视化结果，我们能够直观地观察到信号在频域中的分布特性，为后续分析奠定了基础。

#### 振幅谱与相位谱重要性的实验设计与分析

为了判断在信号从傅里叶变换恢复的过程中，振幅谱和相位谱哪个更重要，我们设计并实现了`ifft_reconstruction_experiment`函数。该函数对图像进行 FFT，分离振幅谱和相位谱，然后通过逆 FFT（`ifft2`、`ifftshift`）进行两种重建实验：

1. 将原始图像的振幅谱与随机噪声的相位谱结合。
2. 将随机噪声的振幅谱与原始图像的相位谱结合。

实验结果通过一个 2x2 的网格图（`./outputs/ifft_reconstruction.png`）展示，包括原始图像、随机噪声图像、仅振幅重建图像和仅相位重建图像。同时，计算了原始图像与两种重建图像之间的均方误差（MSE），分别记录为`ifft_result["mse_amplitude_only"]`和`ifft_result["mse_phase_only"]`。

观察结果表明，尽管仅振幅重建的 MSE 可能较低，但仅相位重建的视觉质量通常更接近原始图像的结构。这是因为相位谱编码了信号分量的空间排列和结构信息（如边缘和轮廓），对视觉感知至关重要。而振幅谱主要编码了频率上的能量分布，缺乏空间信息。因此，保留相位信息通常对保持图像的可识别特征更为关键，即使像 MSE 这样的逐像素误差指标可能更倾向于振幅。

#### 傅里叶频域滤波实现特定效果

在傅里叶频域中，我们通过`filter_audio_fft`函数对音频信号进行 FFT，根据选择的滤波器类型（`lowpass`、`highpass`、`bandpass`、`bandstop`、`notch`）创建频率掩码，在频域中应用掩码，然后通过 IFFT（`ifft`）获取滤波后的音频。

实验结果生成了对比图表（`./outputs/audio_filter_comparison_{filter_type}.png`），展示了原始波形和滤波后波形及其各自的频谱，并保存了滤波后的音频文件（`./outputs/filtered_{filter_type}_{audio_file_name}`）。不同类型的滤波器产生了不同的效果，例如低通滤波使音频听起来沉闷，高通滤波则使其听起来尖锐。通过这些实验，我们实现了类似卷积实践中的特定效果。

---

### 任务 1.2：自定义不同尺寸的输入和卷积核

本任务通过在时（空间）域和频域中实现卷积操作，比较两者的结果差异和运算速度，分析输入和卷积核尺寸对速度的影响及其原因。

#### 实现方法与比较

用户可以通过滑动条（`signal_length_slider`、`kernel_size_slider`、`image_height_slider`等）定义信号/图像尺寸和卷积核大小。代码实现了两种卷积方法：

-   `spatial_convolution`函数在时/空间域直接执行卷积操作，使用`scipy.signal.convolve`或`convolve2d`。
-   `frequency_domain_convolution`函数通过 FFT 方法实现卷积：对信号和卷积核进行 FFT（适当填充以实现“完整”卷积），在频域中进行逐元素相乘，然后进行 IFFT。`apply_fft`和`apply_ifft`包装函数处理库选择。

`compare_spatial_freq_conv`函数协调比较，多次运行两种方法（`run_count_input`），并使用`@timer`装饰器计算平均执行时间。

#### 结果与分析

实验结果输出了空间域和频域卷积的平均执行时间（`practice_1_task_2_result`），计算了速度比（空间域时间/频域时间），并计算了两种方法结果之间的最大和平均绝对差异（应非常接近于零，仅因数值精度而异）。此外，生成了可视化图表（`./outputs/compare_conv.png`），展示了原始信号/图像、卷积核（针对二维）、两种卷积方法的结果、差异图（针对二维）以及执行时间的柱状图对比。

通过调整输入滑动条，可以观察到尺寸对速度的影响：

-   对于较小的信号/图像和卷积核，空间域卷积通常更快，因为 FFT 计算存在开销。
-   随着信号/图像尺寸和/或卷积核尺寸的增加，FFT-based 卷积的 O(N log N)复杂度显著优于直接空间域卷积的 O(N\*M)复杂度（其中 N 为信号大小，M 为卷积核大小）。频域方法在大尺寸输入下显示出明显的速度优势。

---

## 实践二：基于傅里叶变换的复数神经网络

### 任务 2.1：搭建神经网络模型

本任务针对 MNIST 数字分类任务，实现模型的训练和测试，并根据个人电脑硬件情况及对神经网络的了解程度可选其他任务。

#### 实现与结果

-   **数据加载**：使用`torchvision.datasets.MNIST`加载 MNIST 数据集，通过`DataLoader`管理批次（`batchsize_select`），对数据进行归一化处理。提供了样本、分布和批次统计的可视化（`mnist_visualization_dashboard`）。
-   **模型**：定义了一个标准的`SimpleCNN`，使用`torch.nn`，包含三个带 ReLU 激活的`Conv2d`层和一个用于分类的`Linear`层。
-   **训练**：`simple_train`函数实现训练循环，使用 Adam 优化器（`optim.Adam`）和交叉熵损失（`nn.CrossEntropyLoss`），跟踪并绘制训练进度（损失）。
-   **测试**：`simple_test`函数在测试集上评估训练好的模型，使用`calculate_accuracy`辅助函数计算准确率。

用户可以通过`simple_train_button`启动训练，通过`simple_test_button`启动测试。训练过程中打印并绘制训练损失，测试后打印测试准确率（例如“Phy accuracy of the network on 10000 test images: XX.XXX%”）。

---

### 任务 2.2：傅里叶变换与复数神经网络

本任务通过将输入进行傅里叶变换后裁取特定区域作为神经网络输入，基于实数运算实现复数运算，搭建简单的复数神经网络，并分析裁剪区域范围对任务精度的影响。

#### 输入傅里叶变换与区域裁取

在`complex_train`和`complex_test`函数中，输入图像（`inputs`）首先被转换到傅里叶域：使用`np.fft.fft2`后接`np.fft.fftshift`。`crop_fourier_domain`函数根据`crop_region`（中心、左上等）和`crop_size_input`从平移后的傅里叶频谱中选择一个矩形区域。这个裁剪后的复数值张量成为复数网络的输入。可视化单元（`display_cropped_images`）展示了空间域（在白色图像上作为阴影区域）和傅里叶域（在对数振幅谱上作为轮廓）中选择的裁剪区域。

#### 复数神经网络的搭建

定义了一个`ComplexCNN`模型，通过分别处理实部和虚部使用标准 PyTorch 层实现复数层：

-   `ComplexLinear`：实现复数矩阵乘法`W*x`，使用两个实数权重矩阵（`real_weight`、`imag_weight`），执行`(Wr*xr - Wi*xi) + j*(Wr*xi + Wi*xr)`。
-   `ComplexReLU`：对复数的幅度应用 ReLU，同时保留其相位：`ReLU(|z|) * (z / |z|)`。
-   初始复数卷积使用两个实数`Conv2d`层（`conv1_real`、`conv1_imag`）模拟，作为复数核的实部和虚部，根据复数乘法规则应用。

最终输出是最后一个`ComplexLinear`层的复数输出的幅度。

#### 模型训练与测试

-   `complex_train`：类似于`simple_train`，但处理裁剪后的傅里叶域输入并使用`ComplexCNN`。损失（`nn.CrossEntropyLoss`）基于网络的幅度输出计算。
-   `complex_test`：类似于`simple_test`，通过 FFT-裁剪流程处理测试数据并评估`ComplexCNN`，返回准确率。

用户可以通过`complex_train_button`和`complex_test_button`训练和测试复数网络，显示训练损失。结果表（`complex_net_results`，由`display_complex_net_result`显示）记录不同运行的结果，包括轮数、裁剪区域、裁剪大小、最终训练损失和测试准确率。

#### 裁剪区域范围对任务精度的影响分析

通过多次运行训练/测试循环，改变`crop_region`和`crop_size_input`值，观察结果表（`complex_net_results`）中的`Test Acc`，用户可以分析影响：

-   **裁剪大小**：通常，非常小的裁剪大小可能会丢弃过多信息（特别是居中时，去除高频），导致准确率较低。较大的裁剪大小保留更多信息，但增加了网络的输入维度，可能存在一个最佳范围。
-   **裁剪区域**：居中裁剪主要保留低频信息，捕捉粗糙形状；裁剪角落保留更多高频信息（相对于中心），捕捉细节和边缘。低频与高频的重要性取决于任务（MNIST 数字高度依赖形状，因此中心低频至关重要）。比较中心裁剪与角落裁剪的准确率可以揭示哪些频带对`ComplexCNN`在此特定任务上更有信息量。

---
