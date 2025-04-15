import marimo

__generated_with = "0.12.8"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    # 标题部分
    lab_title = r"""
    ## Lab2: 傅里叶变换 Walk-through Guide
    """

    # 实践一的标题
    practice_1_title = r"""
    ### 实践一：傅里叶变换的频谱特性
    """

    # 实践一的第一个任务
    practice_1_task_1 = r"""
    1. **选择喜欢的音乐或图片，实现如下任务：**

           - 在Python平台上，调用现有库函数，实现傅里叶变换。库的选择不限于`scipy`、`numpy`、`torch`等，均包含傅里叶变换、傅里叶反变换的函数。生成并观察傅里叶变换的振幅谱和相位谱。
           - 设计实验方案，判断在信号从傅里叶变换恢复的过程中，振幅谱和相位谱哪个更重要。分析原因。
           - 在傅里叶频域进行特定频段滤波，实现特定效果（如卷积实践中实现的效果）。
    """

    # 实践一的第二个任务
    practice_1_task_2 = r"""
    2. **自定义不同尺寸的输入和卷积核**

           - 在时（空间）域，实现卷积操作
           - 在频域，通过对输入和卷积核分别进行傅里叶变换、频域相乘、傅里叶反变换，实现对应于时（空间）域卷积操作的相同效果
           - 比较二者结果的差别和运算速度的差别，分析输入和卷积核的尺寸对速度的影响及原因

           > 注：可使用`python`的`time`模块，记录运算时间。
    """

    # 实践二的标题
    practice_2_title = r"""
    ### 实践二：基于傅里叶变换的复数神经网络
    """

    # 实践二的第一个任务
    practice_2_task_1 = r"""
    1. **搭建神经网络模型**

           - 针对MNIST数字分类任务，实现模型的训练和测试
           - 或结合个人电脑硬件情况及对神经网络了解程度，任选任务

            > 注：已提供数字分类的参考代码，参考`main.ipynb`。
    """

    # 实践二的第二个任务
    practice_2_task_2 = r"""
    2. **傅里叶变换与复数神经网络**

           - 将输入进行傅里叶变换后，任意裁取区域（注意`fftshift`和`ifftshift`操作）作为神经网络的输入
           - 基于实数运算实现的复数运算，搭建简单的复数神经网络，兼容复数运算
           - 实现模型的训练和测试
           - 分析裁剪区域的范围对任务精度的影响

           > 注：已提供数字分类的参考代码，空间域裁剪代码参考`main_crop.ipynb`，傅里叶域裁剪代码参考`main_crop_fft.ipynb`。该参考代码指定了以图像中心点为裁剪区域的中心点，同学们可修改该处代码，调整为任意裁剪区域。该参考代码默认在CPU上运行，有条件的同学可更改为GPU运行，加快速度。
    """

    # 最后组合所有内容
    lab_content = lab_title + practice_1_title + practice_1_task_1 + practice_1_task_2 + practice_2_title + practice_2_task_1 + practice_2_task_2

    # 在mo.md中显示
    mo.md(text=lab_content)
    return (
        lab_content,
        lab_title,
        practice_1_task_1,
        practice_1_task_2,
        practice_1_title,
        practice_2_task_1,
        practice_2_task_2,
        practice_2_title,
    )


@app.cell(hide_code=True)
def _(mo, seed_input, select_fft_ifft_module):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    from tqdm import tqdm
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import numpy as np
    from pydub import AudioSegment
    from skimage import io, color
    import os
    from io import BytesIO
    from scipy import signal
    from typing import Tuple, List, Union, Optional
    import functools
    import time
    import scipy
    import random
    from datetime import datetime


    # 下面是一些全局变量的定义，你应该不需要修改它们
    OUTPUT_DIR = "./outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    IMAGE_SIZE = 28
    SEED = seed_input.value
    # SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 如果使用GPU，设置CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)


    mo.md(
        text=f"""
    ## 环境设置

    上面的代码块导入了本项目所需的所有第三方库，包括PyTorch、torchvision、matplotlib等用于深度学习和数据处理的库。然后设置一下随机种子。

    {select_fft_ifft_module}

    {seed_input}

    如果遇到`ModuleNotFoundError`错误，请检查并安装缺失的库。

    > 特别注意：如果出现`No module named 'pyaudioop'`错误，这是因为Python 3.13已移除audioop模块，而pydub库依赖它。解决方法是将Python降级到3.12或更早版本，或使用pydub的替代方案。详见[相关issue](https://github.com/jiaaro/pydub/issues/827#issuecomment-2612912859)。

    至此，我们已经完成了基本的准备工作，从右边开始实践一。下面是一些额外的辅助函数，你不需要处理它们。
    """
    )
    return (
        AudioSegment,
        BytesIO,
        IMAGE_SIZE,
        List,
        OUTPUT_DIR,
        Optional,
        SEED,
        Tuple,
        Union,
        color,
        datetime,
        functools,
        io,
        nn,
        np,
        optim,
        os,
        plt,
        random,
        scipy,
        signal,
        time,
        torch,
        torchvision,
        tqdm,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    select_fft_ifft_module = mo.ui.dropdown(options=["PyTorch", "numpy", "scipy"], value="scipy", label="请选择使用哪一个库的`fft/ifft`算法")
    seed_input = mo.ui.number(value=42, label="请设置随机种子", start=1, stop=10000, step=1)
    return seed_input, select_fft_ifft_module


@app.cell(hide_code=True)
def _(time):
    def timer(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            wrapper.last_time = execution_time
            return result
        wrapper.last_time = 0
        return wrapper
    return (timer,)


@app.cell(hide_code=True)
def _(np, scipy, select_fft_ifft_module, torch):
    def get_fft_functions():
        module = select_fft_ifft_module.value

        if module == "scipy":
            return {
                "fft": scipy.fft.fft,
                "ifft": scipy.fft.ifft,
                "fft2": scipy.fft.fft2,
                "ifft2": scipy.fft.ifft2,
                "fftshift": scipy.fft.fftshift,
                "ifftshift": scipy.fft.ifftshift,
                "name": "SciPy"
            }
        elif module == "numpy":
            return {
                "fft": np.fft.fft,
                "ifft": np.fft.ifft,
                "fft2": np.fft.fft2,
                "ifft2": np.fft.ifft2,
                "fftshift": np.fft.fftshift,
                "ifftshift": np.fft.ifftshift,
                "name": "NumPy"
            }
        elif module == "pytorch":
            return {
                "fft": torch.fft.fft,
                "ifft": torch.fft.ifft,
                "fft2": torch.fft.fft2,
                "ifft2": torch.fft.ifft2,
                "fftshift": torch.fft.fftshift,
                "ifftshift": torch.fft.ifftshift,
                "name": "PyTorch"
            }
        else:
            raise ValueError(f"未知的 FFT 模块: {module}")

    fft_funcs = get_fft_functions()

    fft = fft_funcs["fft"]
    ifft = fft_funcs["ifft"]
    fft2 = fft_funcs["fft2"]
    ifft2 = fft_funcs["ifft2"]
    fftshift = fft_funcs["fftshift"]
    ifftshift = fft_funcs["ifftshift"]

    def apply_fft(data, dim=1):
        """应用适当的 FFT 函数，处理数据类型转换

        Args:
            data: 输入数据
            dim: 维度，1 表示一维 FFT，2 表示二维 FFT
        """
        if not isinstance(data, (np.ndarray, torch.Tensor)):
            data = np.array(data)

        if select_fft_ifft_module.value == "pytorch" and not torch.is_tensor(data):
            # 将数据转换为 PyTorch 张量
            tensor_data = torch.tensor(data, dtype=torch.float32)
            if dim == 1:
                result = fft_funcs["fft"](tensor_data)
            else:
                result = fft_funcs["fft2"](tensor_data)
            # 转换回 NumPy（如果需要）
            return result.cpu().numpy() if hasattr(result, 'cpu') else result
        else:
            # 对于非 PyTorch 库或已经是张量的数据，直接应用函数
            if dim == 1:
                return fft_funcs["fft"](data)
            else:
                return fft_funcs["fft2"](data)

    def apply_ifft(data, dim=1):
        """应用适当的 IFFT 函数，处理数据类型转换

        Args:
            data: 输入数据
            dim: 维度，1 表示一维 IFFT，2 表示二维 IFFT
        """
        if not isinstance(data, (np.ndarray, torch.Tensor)):
            data = np.array(data)

        if select_fft_ifft_module.value == "pytorch" and not torch.is_tensor(data):
            # 将数据转换为 PyTorch 张量
            tensor_data = torch.tensor(data, dtype=torch.complex64)
            if dim == 1:
                result = fft_funcs["ifft"](tensor_data)
            else:
                result = fft_funcs["ifft2"](tensor_data)
            # 转换回 NumPy（如果需要）
            return result.cpu().numpy() if hasattr(result, 'cpu') else result
        else:
            # 对于非 PyTorch 库或已经是张量的数据，直接应用函数
            if dim == 1:
                return fft_funcs["ifft"](data)
            else:
                return fft_funcs["ifft2"](data)
    return (
        apply_fft,
        apply_ifft,
        fft,
        fft2,
        fft_funcs,
        fftshift,
        get_fft_functions,
        ifft,
        ifft2,
        ifftshift,
    )


@app.cell(column=1, hide_code=True)
def _(mo, practice_1_task_1, practice_1_title):
    mo.md(text=practice_1_title + practice_1_task_1)
    return


@app.cell(hide_code=True)
def _(OUTPUT_DIR, audio_file_upload, mo, os, run_1d_fft_button):
    mo.md(
        text=f"""
        ### 一维信号傅里叶变换可视化

        {mo.vstack([audio_file_upload, run_1d_fft_button])}

        可视化结果：
        {mo.image(src=f"./{OUTPUT_DIR}/audio_fft_analysis.png", caption="一维信号傅里叶变换可视化结果") if os.path.exists(f"./{OUTPUT_DIR}/audio_fft_analysis.png") else "请先运行"}
        """
    )
    return


@app.cell
def _(AudioSegment, BytesIO, OUTPUT_DIR, fft, fftshift, np, os, plt):
    def fft_1d(audio_file_name, audio_bytes_data, output_dir=OUTPUT_DIR):
        """
        Perform 1D FFT on audio data and visualize results.

        Parameters:
        -----------
        audio_file : str
            Path to the audio file
        output_dir : str
            Directory to save visualization results
        """
        # Load audio file
        audio_file_type = os.path.splitext(audio_file_name)[1][1:].lower()

        # If extension is empty or not recognized, default to wav
        if not audio_file_type or audio_file_type not in ['wav', 'mp3', 'ogg', 'flac']:
            audio_file_type = 'wav'

        # Load audio file
        # audio = AudioSegment.from_mp3(audio_file)
        audio = AudioSegment.from_file(BytesIO(audio_bytes_data), format=audio_file_type)
        audio_data = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
        audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize

        # Perform FFT on audio data
        audio_fft = fftshift(fft(audio_data))
        audio_magnitude = np.abs(audio_fft)
        audio_phase = np.angle(audio_fft)

        # Visualize audio and its FFT
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Original audio waveform
        axs[0].plot(audio_data)
        axs[0].set_title(f'Audio "{audio_file_name}" Waveform')

        # Audio magnitude spectrum
        axs[1].plot(audio_magnitude)
        axs[1].set_title('Audio Magnitude Spectrum')

        # Audio phase spectrum
        axs[2].plot(audio_phase)
        axs[2].set_title('Audio Phase Spectrum')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'audio_fft_analysis.png'), dpi=300)
        plt.close()

        return audio_data, audio_magnitude, audio_phase, sample_rate
    return (fft_1d,)


@app.cell
def _(OUTPUT_DIR, image_file_upload, mo, os, run_2d_fft_button):
    mo.md(
        text=f"""
        ### 二维信号傅里叶变换可视化

        {mo.vstack([image_file_upload, run_2d_fft_button])}

        可视化结果：
        {mo.image(src=f"./{OUTPUT_DIR}/image_fft_analysis.png", caption="二维信号傅里叶变换可视化结果") if os.path.exists(f"./{OUTPUT_DIR}/image_fft_analysis.png") else "请先运行"}
        """
    )
    return


@app.cell
def _(BytesIO, OUTPUT_DIR, color, fft, fftshift, io, np, os, plt):
    def fft_2d(image_file_name, image_bytes_data, output_dir=OUTPUT_DIR):
        """
        Perform 2D FFT on image data and visualize results.

        Parameters:
        -----------
        image_file_name : str
            Name of the image file
        image_bytes_data : bytes
            Image data as bytes
        output_dir : str
            Directory to save visualization results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load image from bytes
        image_stream = BytesIO(image_bytes_data)
        image = io.imread(image_stream)

        if image.ndim == 3:
            image = color.rgb2gray(image)  # Convert to grayscale

        # Perform 2D FFT on image
        image_fft = fftshift(fft(fft(image, axis=0), axis=1))
        image_magnitude = np.abs(image_fft)
        image_phase = np.angle(image_fft)

        # Visualize image and its FFT
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axs[0].imshow(image, cmap='gray')
        axs[0].set_title(f'"{os.path.basename(image_file_name)}" Image')
        axs[0].axis('off')

        # Image magnitude spectrum
        axs[1].imshow(np.log(1 + image_magnitude), cmap='gray')
        axs[1].set_title('Image Magnitude Spectrum')
        axs[1].axis('off')

        # Image phase spectrum
        axs[2].imshow(image_phase, cmap='gray')
        axs[2].set_title('Image Phase Spectrum')
        axs[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'image_fft_analysis.png'), dpi=300)
        plt.close()

        return image, image_magnitude, image_phase
    return (fft_2d,)


@app.cell
def _(OUTPUT_DIR, ifft_result, image_file_upload, mo, os, run_ifft_button):
    mo.md(
        text=f"""
    ### 图像傅里叶变换重构实验

    {mo.vstack([image_file_upload, run_ifft_button])}

    可视化结果：
    {mo.image(src=f"./{OUTPUT_DIR}/ifft_reconstruction.png", caption="振幅谱与相位谱重要性对比实验") if os.path.exists(f"./{OUTPUT_DIR}/ifft_reconstruction.png") else "请先运行实验"}

    谁（振幅谱/相位谱）更重要？假如我们根据 `MSE` (原图和重构后图片的均方差)衡量: 

        - 仅保留振幅谱的重构与原图的均方误差: {ifft_result.get("mse_amplitude_only", "请先运行代码")}

        - 仅保留相位谱的重构与原图的均方误差: {ifft_result.get("mse_phase_only", "请先运行代码")}

        - 可以看到{"相位谱对图像重构更为重要" if ifft_result.get("mse_phase_only", "请先运行代码") < ifft_result.get("mse_amplitude_only", "请先运行代码") else "振幅谱对图像重构更为重要"}

    但是肉眼看上面的可视化结果，明明是相位谱重构的效果更好，更接近原图，这是为什么呢？
        """
    )
    return


@app.cell
def _(mo):
    mo.md(text=f"""
    振幅谱/相位谱究竟谁更重要？我认为这个取决于我们怎么理解“更重要”的含义。在上面的实践中，我们用`MSE`来作为评判标准的话，结论就会是振幅谱更重要。振幅谱重构保留了图像的整体亮度分布，虽然结构信息丢失严重，但在像素均值层面上误差可能较小。

    这个其实也是很好理解的。想象一个更简单的信号：正弦信号。我们通过把信号强度全部设置成0来改变振幅，这个只能产生一份误差；而如果我们通过讲信号延迟半个周期来改变相位，这个就会产生两份误差。

    {mo.image(src="./demo/ifft_reconstruction_example.png")}

    回到二维信号（图像重构）的例子上，图像中的边缘是由多个频率成分以特定相位关系叠加形成的。只要相位保持不变，人眼更容易识别轮廓。也因此从可视化的结果看，我们会觉得相位谱恢复效果更好，更加重要。
    """)
    return


@app.cell
def _(
    BytesIO,
    OUTPUT_DIR,
    color,
    fft,
    fftshift,
    ifft2,
    ifftshift,
    io,
    np,
    os,
    plt,
):
    def ifft_reconstruction_experiment(image_file_name, image_bytes_data, output_dir=OUTPUT_DIR):
        """
        Experiment to compare the importance of amplitude and phase spectra in image reconstruction
        using inverse Fourier transform (IFFT).

        Parameters:
        -----------
        image_file_name : str
            Image file name
        image_bytes_data : bytes
            Image data in bytes format
        output_dir : str
            Output directory for saving results

        Returns:
        -----------
        dict : Dictionary containing original image, amplitude spectrum, phase spectrum, and reconstructed images
        """

        # Load image from bytes
        image_stream = BytesIO(image_bytes_data)
        image = io.imread(image_stream)

        # Convert to grayscale if colored
        if image.ndim == 3:
            image = color.rgb2gray(image)

        # Perform 2D FFT on image
        image_fft = fftshift(fft(fft(image, axis=0), axis=1))

        # Extract amplitude and phase spectra
        amplitude_spectrum = np.abs(image_fft)
        phase_spectrum = np.angle(image_fft)

        # Create random noise image for comparison
        random_image = np.random.rand(*image.shape)
        random_image_fft = fftshift(fft(fft(random_image, axis=0), axis=1))
        random_amplitude = np.abs(random_image_fft)
        random_phase = np.angle(random_image_fft)

        # Experiment 1: Keep original amplitude, use random phase
        hybrid_fft1 = amplitude_spectrum * np.exp(1j * random_phase)
        amplitude_only_reconstruction = np.real(ifft2(ifftshift(hybrid_fft1)))

        # Experiment 2: Keep original phase, use random amplitude
        hybrid_fft2 = random_amplitude * np.exp(1j * phase_spectrum)
        phase_only_reconstruction = np.real(ifft2(ifftshift(hybrid_fft2)))

        # Visualize results in a 2x2 grid
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        # Original image
        axs[0, 0].imshow(image, cmap='gray')
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')

        # Random noise image
        axs[0, 1].imshow(random_image, cmap='gray')
        axs[0, 1].set_title('Random Noise Image')
        axs[0, 1].axis('off')

        # Amplitude-only reconstruction
        axs[1, 0].imshow(amplitude_only_reconstruction, cmap='gray')
        axs[1, 0].set_title('Amplitude-only Reconstruction')
        axs[1, 0].axis('off')

        # Phase-only reconstruction
        axs[1, 1].imshow(phase_only_reconstruction, cmap='gray')
        axs[1, 1].set_title('Phase-only Reconstruction')
        axs[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ifft_reconstruction.png"), dpi=300)
        plt.close()

        # Calculate mean squared error
        mse_amplitude_only = np.mean((image - amplitude_only_reconstruction) ** 2)
        mse_phase_only = np.mean((image - phase_only_reconstruction) ** 2)

        # Return results
        return {
            "original_image": image,
            "random_image": random_image,
            "amplitude_spectrum": amplitude_spectrum,
            "phase_spectrum": phase_spectrum,
            "amplitude_only_reconstruction": amplitude_only_reconstruction,
            "phase_only_reconstruction": phase_only_reconstruction,
            "mse_amplitude_only": mse_amplitude_only,
            "mse_phase_only": mse_phase_only,
        }
    return (ifft_reconstruction_experiment,)


@app.cell
def _(
    OUTPUT_DIR,
    audio_file_upload,
    filter_descriptions,
    mo,
    run_audio_filter_button,
    select_filter_type,
):
    mo.md(text=f"""
    ### 傅里叶频域滤波实验

    在傅里叶频域进行特定频段滤波，实现特定效果（如卷积实践中实现的效果）。

    我们这里只实现了对音频滤波的代码。

    {mo.vstack([audio_file_upload, mo.hstack([select_filter_type, run_audio_filter_button])])}

    {filter_descriptions[select_filter_type.value]}

    {mo.vstack([mo.image(src=f"./{OUTPUT_DIR}/audio_filter_comparison_{select_filter_type.value}.png"), 
    mo.audio(src=f"./{OUTPUT_DIR}/filtered_{select_filter_type.value}_{audio_file_upload.name()}")])}
    """
    )
    return


@app.cell
def _(AudioSegment, BytesIO, OUTPUT_DIR, fft, fftshift, ifft, np, os, plt):
    def filter_audio_fft(audio_file_name, audio_bytes_data, filter_type='lowpass', output_dir=OUTPUT_DIR):
        """
        Perform 1D FFT on audio data, apply specified filter, and save the filtered audio.

        Parameters:
        -----------
        audio_file_name : str
            Name of the audio file
        audio_bytes_data : bytes
            Audio file data in bytes format
        filter_type : str
            Type of filter to apply. Options: 'lowpass', 'highpass', 'bandpass', 'bandstop', 'notch'
        output_dir : str
            Directory to save visualization results and filtered audio

        Returns:
        --------
        dict
            Dictionary containing paths to the original and filtered audio files, and visualization
        """

        # Load audio file
        audio_file_type = os.path.splitext(audio_file_name)[1][1:].lower()

        # If extension is empty or not recognized, default to wav
        if not audio_file_type or audio_file_type not in ['wav', 'mp3', 'ogg', 'flac']:
            audio_file_type = 'wav'

        audio = AudioSegment.from_file(BytesIO(audio_bytes_data), format=audio_file_type)
        audio_data = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
        channels = audio.channels

        # Handle stereo audio (process each channel separately)
        if channels == 2:
            # Reshape for stereo processing
            audio_data = audio_data.reshape(-1, 2)
            left_channel = audio_data[:, 0]
            right_channel = audio_data[:, 1]

            # Process each channel
            filtered_left = apply_filter(left_channel, filter_type, sample_rate)
            filtered_right = apply_filter(right_channel, filter_type, sample_rate)

            # Combine channels
            filtered_audio_data = np.column_stack((filtered_left, filtered_right)).flatten()
        else:
            # Process mono audio
            filtered_audio_data = apply_filter(audio_data, filter_type, sample_rate)

        # Convert filtered data back to audio
        filtered_audio_data = np.int16(filtered_audio_data * 32767)  # Convert back to 16-bit PCM

        # Create a new AudioSegment from the filtered data
        filtered_audio = AudioSegment(
            filtered_audio_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit audio
            channels=channels
        )

        # Save filtered audio
        filtered_audio_path = os.path.join(output_dir, f'filtered_{filter_type}_{os.path.basename(audio_file_name)}')
        filtered_audio.export(filtered_audio_path, format=audio_file_type)

        # Visualize original and filtered audio
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Original audio waveform
        axs[0, 0].plot(audio_data if channels == 1 else audio_data[:, 0])
        axs[0, 0].set_title(f'Original Audio Waveform')

        # Original audio spectrum
        original_fft = np.abs(fftshift(fft(audio_data if channels == 1 else audio_data[:, 0])))
        freq = np.linspace(-sample_rate/2, sample_rate/2, len(original_fft))
        axs[0, 1].plot(freq, original_fft)
        axs[0, 1].set_title('Original Audio Spectrum')
        axs[0, 1].set_xlabel('Frequency (Hz)')

        # Filtered audio waveform
        filtered_display = filtered_audio_data if channels == 1 else filtered_audio_data.reshape(-1, 2)[:, 0]
        axs[1, 0].plot(filtered_display)
        axs[1, 0].set_title(f'Filtered Audio Waveform ({filter_type})')

        # Filtered audio spectrum
        filtered_fft = np.abs(fftshift(fft(filtered_display)))
        axs[1, 1].plot(freq, filtered_fft)
        axs[1, 1].set_title(f'Filtered Audio Spectrum ({filter_type})')
        axs[1, 1].set_xlabel('Frequency (Hz)')

        plt.tight_layout()
        viz_path = os.path.join(output_dir, f'audio_filter_comparison_{filter_type}.png')
        plt.savefig(viz_path, dpi=300)
        plt.close()

        return {
            "original_audio": audio_file_name,
            "filtered_audio": filtered_audio_path,
            "visualization": viz_path,
            "filter_type": filter_type
        }

    def apply_filter(audio_data, filter_type, sample_rate):
        """
        Apply the specified filter to audio data in the frequency domain.

        Parameters:
        -----------
        audio_data : numpy.ndarray
            Audio data to filter
        filter_type : str
            Type of filter to apply
        sample_rate : int
            Sample rate of the audio

        Returns:
        --------
        numpy.ndarray
            Filtered audio data
        """
        # Normalize audio data
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Perform FFT
        audio_fft = fft(audio_data)
        n = len(audio_fft)

        # Create frequency bins
        freq = np.fft.fftfreq(n, d=1/sample_rate)

        # Create filter mask based on filter type
        mask = np.ones(n, dtype=complex)

        if filter_type == 'lowpass':
            # Low-pass filter: keep frequencies below 1000 Hz
            cutoff = 1000
            mask[np.abs(freq) > cutoff] = 0

        elif filter_type == 'highpass':
            # High-pass filter: keep frequencies above 2000 Hz
            cutoff = 2000
            mask[np.abs(freq) < cutoff] = 0

        elif filter_type == 'bandpass':
            # Band-pass filter: keep frequencies between 500 and 2000 Hz
            low_cutoff = 500
            high_cutoff = 2000
            mask[(np.abs(freq) < low_cutoff) | (np.abs(freq) > high_cutoff)] = 0

        elif filter_type == 'bandstop':
            # Band-stop filter: remove frequencies between 500 and 2000 Hz
            low_cutoff = 500
            high_cutoff = 2000
            mask[(np.abs(freq) >= low_cutoff) & (np.abs(freq) <= high_cutoff)] = 0

        elif filter_type == 'notch':
            # Notch filter: remove a narrow band around 1000 Hz
            center = 1000
            width = 100
            mask[(np.abs(freq) >= center - width/2) & (np.abs(freq) <= center + width/2)] = 0

        # Apply filter in frequency domain
        filtered_fft = audio_fft * mask

        # Convert back to time domain
        filtered_audio = np.real(ifft(filtered_fft))

        return filtered_audio
    return apply_filter, filter_audio_fft


@app.cell(hide_code=True)
def _(mo):
    mo.md(text="""
    以上就是全部的实践一任务一的内容，下面是一些辅助函数，你不需要处理。
    """)
    return


@app.cell(hide_code=True)
def _(audio_file_upload, fft_1d, run_1d_fft_button):
    if run_1d_fft_button.value:
        _ = fft_1d(audio_file_name=audio_file_upload.name(), audio_bytes_data=audio_file_upload.contents())
    return


@app.cell(hide_code=True)
def _(fft_2d, image_file_upload, run_2d_fft_button):
    if run_2d_fft_button.value:
        _ = fft_2d(image_file_name=image_file_upload.name(), image_bytes_data=image_file_upload.contents())
    return


@app.cell(hide_code=True)
def _(mo):
    audio_file_upload = mo.ui.file(kind="area")
    run_1d_fft_button = mo.ui.run_button(label="1D fft")

    image_file_upload = mo.ui.file(kind="area")
    run_2d_fft_button = mo.ui.run_button(label="2D fft")

    select_filter_type = mo.ui.dropdown(options=["lowpass", "highpass", "bandpass", "bandstop", "notch"], label="选择滤波类型", allow_select_none=False, value="lowpass")
    run_audio_filter_button = mo.ui.run_button(label="Click me! 对音频滤波")

    run_ifft_button = mo.ui.run_button(label="ifft：从傅里叶变换恢复")
    return (
        audio_file_upload,
        image_file_upload,
        run_1d_fft_button,
        run_2d_fft_button,
        run_audio_filter_button,
        run_ifft_button,
        select_filter_type,
    )


@app.cell(hide_code=True)
def _(ifft_reconstruction_experiment, image_file_upload, run_ifft_button):
    ifft_result = {}
    if run_ifft_button.value:
        ifft_result = ifft_reconstruction_experiment(image_file_name=image_file_upload.name(), image_bytes_data=image_file_upload.contents())
    return (ifft_result,)


@app.cell(hide_code=True)
def _():
    filter_descriptions = {
            'lowpass': "低通滤波：保留低频成分，去除高频成分，使声音听起来更加沉闷、温暖，类似于隔墙听声音的效果。",
            'highpass': "高通滤波：保留高频成分，去除低频成分，声音会变得尖锐清晰，但缺少低音，像电话通话的声音。",
            'bandpass': "带通滤波：只保留特定频率范围内的声音，其他频率被过滤，类似于通过特定管道传播的声音。",
            'bandstop': "带阻滤波：去除特定频率范围内的声音，可用于消除特定噪声，如电源嗡嗡声。",
            'notch': "陷波滤波：精确去除非常窄的频率范围，通常用于消除单一频率的干扰，如50/60Hz电源噪声。"
    }
    return (filter_descriptions,)


@app.cell(hide_code=True)
def _(
    audio_file_upload,
    filter_audio_fft,
    run_audio_filter_button,
    select_filter_type,
):
    if run_audio_filter_button.value:
        _ = filter_audio_fft(audio_file_name=audio_file_upload.name(), audio_bytes_data=audio_file_upload.contents(), filter_type=select_filter_type.value)
    return


@app.cell(column=2, hide_code=True)
def _(mo, practice_1_task_2, practice_1_title):
    mo.md(text=practice_1_title + practice_1_task_2)
    return


@app.cell(hide_code=True)
def _(
    OUTPUT_DIR,
    dimension_selector,
    image_height_slider,
    image_width_slider,
    kernel_height_slider,
    kernel_size_slider,
    kernel_width_slider,
    mo,
    os,
    practice_1_task_2_result,
    run_count_input,
    run_practice_1_task_2,
    signal_length_slider,
    visualize_1d_signal,
    visualize_2d_signal,
):
    if dimension_selector.value == "1D":
        fig = visualize_1d_signal(signal_length_slider.value, kernel_size_slider.value)
    else:  # 2D
        fig = visualize_2d_signal(image_height_slider.value, image_width_slider.value, 
                                 kernel_height_slider.value, kernel_width_slider.value)
    _image_path = os.path.join(OUTPUT_DIR, "vis_generated_1d_input.png" if dimension_selector.value == "1D" else "vis_generated_2d_input.png")

    mo.md(text=f"""
    ### 自定义输入

    可以通过下面这些控件生成想要的输入和卷积核

    {dimension_selector}

    {mo.hstack([
        mo.vstack([signal_length_slider, kernel_size_slider]),
        mo.vstack([image_height_slider, image_width_slider, kernel_height_slider, kernel_width_slider])
    ], 
               justify="space-between")
    }

    {mo.image(alt="设置参数观察signal和kernel", src=_image_path) if os.path.exists(_image_path) else "请先设置参数生成输入"}

    ---

    {mo.hstack([run_count_input, run_practice_1_task_2])}

    {"请点击运行按钮开始计算" if practice_1_task_2_result is None else f'''
    空间域卷积时间: {practice_1_task_2_result[0]:.4f}秒

    频域卷积时间: {practice_1_task_2_result[1]:.4f}秒

    加速比: {(practice_1_task_2_result[0] / practice_1_task_2_result[1]):.2f}倍
    '''}

    {mo.image(src=f"{os.path.join(OUTPUT_DIR, 'compare_conv.png')}")}
    """)
    return (fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(text="""
    可以反复使用不同参数，比较不同情境下时（空间）域和傅里叶域卷积操作的运算速度差别。

    下面是辅助函数，不要修改。
    """)
    return


@app.cell(hide_code=True)
def _(frequency_domain_convolution, np, os, plt, spatial_convolution, timer):
    @timer
    def compare_spatial_freq_conv(signal, kernel, dim, count=1, output_dir="output"):
        """
        比较时域卷积和频域卷积的结果和性能

        Args:
            signal: 输入信号/图像
            kernel: 卷积核
            dim: 维度 (1=一维, 2=二维)
            count: 重复执行的次数，用于计算平均时间
            output_dir: 输出图像的目录

        Returns:
            tuple: (spatial_time, freq_time) 时域和频域卷积的执行时间
        """

        # 执行时域卷积多次并计算平均时间
        spatial_times = []
        for _ in range(count):
            spatial_result = spatial_convolution(signal=signal, kernel=kernel, dim=dim)
            spatial_times.append(spatial_convolution.last_time)
        avg_spatial_time = sum(spatial_times) / count

        # 执行频域卷积多次并计算平均时间
        freq_times = []
        for _ in range(count):
            freq_result = frequency_domain_convolution(signal=signal, kernel=kernel, dim=dim)
            freq_times.append(frequency_domain_convolution.last_time)
        avg_freq_time = sum(freq_times) / count

        # 计算结果差异
        diff = np.abs(spatial_result - freq_result)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        # 创建可视化比较图
        plt.figure(figsize=(15, 5) if dim == 1 else (15, 10))

        if dim == 1:
            # 一维信号的可视化
            plt.subplot(1, 3, 1)
            plt.plot(signal)
            plt.title('Original Signal')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(spatial_result)
            plt.title('Spatial Domain Convolution')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.plot(freq_result)
            plt.title('Frequency Domain Convolution')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.grid(True)

        else:
            # 二维图像的可视化
            plt.subplot(2, 3, 1)
            plt.imshow(signal, cmap='viridis')
            plt.title('Original Image')
            plt.colorbar()

            plt.subplot(2, 3, 2)
            plt.imshow(kernel, cmap='viridis')
            plt.title('Convolution Kernel')
            plt.colorbar()

            plt.subplot(2, 3, 3)
            plt.imshow(diff, cmap='hot')
            plt.title(f'Difference (Max: {max_diff:.2e})')
            plt.colorbar()

            plt.subplot(2, 3, 4)
            plt.imshow(spatial_result, cmap='viridis')
            plt.title(f'Spatial Conv. ({avg_spatial_time:.4f}s)')
            plt.colorbar()

            plt.subplot(2, 3, 5)
            plt.imshow(freq_result, cmap='viridis')
            plt.title(f'Frequency Conv. ({avg_freq_time:.4f}s)')
            plt.colorbar()

            plt.subplot(2, 3, 6)
            # 绘制执行时间对比条形图
            methods = ['Spatial', 'Frequency']
            times = [avg_spatial_time, avg_freq_time]
            plt.bar(methods, times, color=['blue', 'orange'])
            plt.title('Execution Time Comparison')
            plt.ylabel('Time (seconds)')
            plt.grid(axis='y')

        plt.tight_layout()
        plt.suptitle(f"Convolution Comparison ({dim}D) - Signal Size: {signal.shape}, Kernel Size: {kernel.shape}", 
                     fontsize=16, y=1.05)

        # 保存图像
        output_path = os.path.join(output_dir, "compare_conv.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

        # 打印比较结果
        print(f"\n--- {dim}D Convolution Comparison ---")
        print(f"Signal shape: {signal.shape}")
        print(f"Kernel shape: {kernel.shape}")
        print(f"Spatial domain convolution time: {avg_spatial_time:.6f}s")
        print(f"Frequency domain convolution time: {avg_freq_time:.6f}s")
        print(f"Speed ratio (spatial/frequency): {avg_spatial_time/avg_freq_time:.2f}x")
        print(f"Maximum difference: {max_diff:.10e}")
        print(f"Mean difference: {mean_diff:.10e}")
        print(f"Output saved to: {output_path}")

        return avg_spatial_time, avg_freq_time
    return (compare_spatial_freq_conv,)


@app.cell(hide_code=True)
def _(apply_fft, apply_ifft, np, scipy, timer):
    # 执行时域卷积的函数
    @timer
    def spatial_convolution(signal, kernel, dim=1):
        """
        在时域/空间域执行卷积

        Args:
            signal: 输入信号/图像
            kernel: 卷积核
            dim: 维度 (1=一维, 2=二维)

        Returns:
            np.ndarray: 卷积结果
        """
        if dim == 1:
            # 一维卷积（使用 scipy.signal.convolve 以获得完整输出）
            return scipy.signal.convolve(signal, kernel, mode='full')
        else:
            # 二维卷积
            return scipy.signal.convolve2d(signal, kernel, mode='full')

    # 执行频域乘积的函数
    @timer
    def frequency_domain_convolution(signal, kernel, dim=1):
        """
        通过频域乘积实现卷积

        Args:
            signal: 输入信号/图像
            kernel: 卷积核
            dim: 维度 (1=一维, 2=二维)

        Returns:
            np.ndarray: 卷积结果
        """
        if dim == 1:
            # 为了匹配 scipy.signal.convolve 的 'full' 模式，需要进行零填充
            n_out = len(signal) + len(kernel) - 1
            signal_padded = np.pad(signal, (0, n_out - len(signal)))
            kernel_padded = np.pad(kernel, (0, n_out - len(kernel)))

            # 计算 FFT
            signal_fft = apply_fft(signal_padded, dim=1)
            kernel_fft = apply_fft(kernel_padded, dim=1)

            # 频域相乘
            result_fft = signal_fft * kernel_fft

            # 反变换回时域
            result = apply_ifft(result_fft, dim=1)

            # 取实部（由于数值误差可能会有微小的虚部）
            return np.real(result)
        else:
            # 为二维情况计算输出大小
            h_out = signal.shape[0] + kernel.shape[0] - 1
            w_out = signal.shape[1] + kernel.shape[1] - 1

            # 零填充
            signal_padded = np.pad(signal, ((0, h_out - signal.shape[0]), (0, w_out - signal.shape[1])))
            kernel_padded = np.pad(kernel, ((0, h_out - kernel.shape[0]), (0, w_out - kernel.shape[1])))

            # 计算 FFT
            signal_fft = apply_fft(signal_padded, dim=2)
            kernel_fft = apply_fft(kernel_padded, dim=2)

            # 频域相乘
            result_fft = signal_fft * kernel_fft

            # 反变换回空间域
            result = apply_ifft(result_fft, dim=2)

            # 取实部
            return np.real(result)
    return frequency_domain_convolution, spatial_convolution


@app.cell(hide_code=True)
def _(OUTPUT_DIR, generate_signal_1d, generate_signal_2d, np, os, plt):
    # 定义1D信号可视化函数
    def visualize_1d_signal(signal_length, kernel_size, output_dir=OUTPUT_DIR):
        # 生成1D信号和卷积核
        signal_1d, kernel_1d = generate_signal_1d(signal_length=signal_length, 
                                            kernel_size=kernel_size)

        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        # 绘制信号
        ax1.plot(np.arange(len(signal_1d)), signal_1d)
        ax1.set_title('Input Signal')
        ax1.set_xlabel('Sample Points')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)

        # 绘制卷积核
        ax2.plot(np.arange(len(kernel_1d)), kernel_1d)
        ax2.set_title('Kernel')
        ax2.set_xlabel('Sample Points')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "vis_generated_1d_input.png"))
        return fig

    # 定义2D图像可视化函数
    def visualize_2d_signal(image_height, image_width, kernel_height, kernel_width, output_dir=OUTPUT_DIR):
        # 生成2D图像和卷积核
        image_2d, kernel_2d = generate_signal_2d(image_size=(image_height, image_width), 
                                                kernel_size=(kernel_height, kernel_width))

        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 绘制图像
        im1 = ax1.imshow(image_2d, cmap='viridis')
        ax1.set_title('Input Image')
        fig.colorbar(im1, ax=ax1, shrink=0.8)

        # 绘制卷积核
        im2 = ax2.imshow(kernel_2d, cmap='viridis')
        ax2.set_title('Kernel')
        fig.colorbar(im2, ax=ax2, shrink=0.8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "vis_generated_2d_input.png"))
        return fig
    return visualize_1d_signal, visualize_2d_signal


@app.cell(hide_code=True)
def _(mo):
    # 创建UI控件
    signal_length_slider = mo.ui.slider(start=128, stop=32768, step=128, value=1024, label="信号长度", show_value=True, debounce=True)
    kernel_size_slider = mo.ui.slider(start=3, stop=1001, step=2, value=31, label="卷积核大小", show_value=True, debounce=True)

    image_height_slider = mo.ui.slider(start=32, stop=1024, step=32, value=128, label="图像高度", show_value=True, debounce=True)
    image_width_slider = mo.ui.slider(start=32, stop=1024, step=32, value=128, label="图像宽度", show_value=True, debounce=True)
    kernel_height_slider = mo.ui.slider(start=3, stop=51, step=2, value=7, label="卷积核高度", show_value=True, debounce=True)
    kernel_width_slider = mo.ui.slider(start=3, stop=51, step=2, value=7, label="卷积核宽度", show_value=True, debounce=True)

    dimension_selector = mo.ui.dropdown(
        options=["1D", "2D"], 
        value="1D", 
        label="选择卷积维度"
    )

    run_count_input = mo.ui.number(debounce=True, start=1, stop=100, step=1, label="重复运行次数", value=1)
    run_practice_1_task_2 = mo.ui.run_button(label="Run")
    return (
        dimension_selector,
        image_height_slider,
        image_width_slider,
        kernel_height_slider,
        kernel_size_slider,
        kernel_width_slider,
        run_count_input,
        run_practice_1_task_2,
        signal_length_slider,
    )


@app.cell(hide_code=True)
def _(
    OUTPUT_DIR,
    compare_spatial_freq_conv,
    dimension_selector,
    generate_signal_1d,
    generate_signal_2d,
    image_height_slider,
    image_width_slider,
    kernel_height_slider,
    kernel_size_slider,
    kernel_width_slider,
    run_count_input,
    run_practice_1_task_2,
    signal_length_slider,
):
    practice_1_task_2_result = None

    if run_practice_1_task_2.value:
        if dimension_selector.value == "1D":
            signal_size = int(signal_length_slider.value)
            kernel_size = int(kernel_size_slider.value)

            _signal, _kernel = generate_signal_1d(signal_length=signal_size, kernel_size=kernel_size)

            practice_1_task_2_result = compare_spatial_freq_conv(
                signal=_signal,
                kernel=_kernel,
                dim=1,
                count=int(run_count_input.value),
                output_dir=OUTPUT_DIR
            )

        else:  # 2D
            image_height = int(image_height_slider.value)
            image_width = int(image_width_slider.value)
            kernel_height = int(kernel_height_slider.value)
            kernel_width = int(kernel_width_slider.value)

            _signal, _kernel = generate_signal_2d(
                image_size=(image_height, image_width),
                kernel_size=(kernel_height, kernel_width)
            )

            practice_1_task_2_result = compare_spatial_freq_conv(
                signal=_signal,
                kernel=_kernel,
                dim=2,
                count=int(run_count_input.value),
                output_dir=OUTPUT_DIR
            )
    return (
        image_height,
        image_width,
        kernel_height,
        kernel_size,
        kernel_width,
        practice_1_task_2_result,
        signal_size,
    )


@app.cell(hide_code=True)
def _(Tuple, np):
    # 生成测试信号和卷积核的函数
    def generate_signal_1d(signal_length: int, kernel_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成一维测试信号和卷积核

        Args:
            signal_length: 信号长度
            kernel_size: 卷积核大小

        Returns:
            Tuple[np.ndarray, np.ndarray]: 信号和卷积核
        """
        # 生成包含多个频率成分的测试信号
        t = np.linspace(0, 1, signal_length)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 20 * t)

        # 生成高斯卷积核
        kernel = np.exp(-0.5 * ((np.arange(kernel_size) - kernel_size // 2) / (kernel_size / 6)) ** 2)
        kernel = kernel / np.sum(kernel)  # 归一化卷积核

        return signal, kernel

    def generate_signal_2d(image_size: Tuple[int, int], kernel_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成二维测试图像和卷积核

        Args:
            image_size: 图像尺寸 (height, width)
            kernel_size: 卷积核尺寸 (height, width)

        Returns:
            Tuple[np.ndarray, np.ndarray]: 图像和卷积核
        """
        # 生成测试图像（包含几何形状和一些噪声）
        image = np.zeros(image_size)
        h, w = image_size

        # 添加几个几何形状
        # 中心方块
        image[h//4:3*h//4, w//4:3*w//4] = 0.5
        # 对角线
        for i in range(min(h, w)):
            if i < min(h, w):
                image[i, i] = 1.0
        # 添加一些高斯噪声
        image += 0.1 * np.random.randn(*image_size)

        # 生成高斯卷积核（模糊）
        kh, kw = kernel_size
        y, x = np.mgrid[-(kh//2):kh//2+1, -(kw//2):kw//2+1]
        kernel = np.exp(-(x**2 + y**2) / (2 * (min(kh, kw) / 4)**2))
        kernel = kernel / np.sum(kernel)  # 归一化卷积核

        return image, kernel
    return generate_signal_1d, generate_signal_2d


@app.cell(column=3, hide_code=True)
def _(mo, practice_2_task_1, practice_2_title):
    mo.md(text=practice_2_title + practice_2_task_1)
    return


@app.cell(hide_code=True)
def _(batchsize_select, mnist_visualization_dashboard, mo):
    mo.md(text=f"""
    ### Step-1 数据集

    我们先加载MNIST数据集，然后从里面随机选取几个看看数据的样子。

    {batchsize_select}

    {mnist_visualization_dashboard()}
    """)
    return


@app.cell
def _(batchsize_select, torch, torchvision, transforms):
    # batchsize 表示一次训练投喂的样本数
    batchsize = batchsize_select.value

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    return batchsize, testloader, testset, trainloader, trainset, transform


@app.cell
def _(IMAGE_SIZE, mo, nn):
    class SimpleCNN(nn.Module):
        def __init__(self, image_size, feature_map_num=16):
            super(SimpleCNN, self).__init__()
            self.image_size = image_size
            self.feature_map_num = feature_map_num
            self.conv1 = nn.Conv2d(1, feature_map_num, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(feature_map_num, feature_map_num, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(feature_map_num, feature_map_num, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(feature_map_num * image_size * image_size, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = x.view(-1, self.feature_map_num * self.image_size ** 2)
            x = self.fc1(x)
            return x

    simple_net = SimpleCNN(image_size=IMAGE_SIZE)
    mo.md(text=f"""
    ### Step-2 定义网络

    定义一个简单的卷积神经网络，包含三个卷积层和一个全连接层。网络的结构如下:
    ```plaintext
    {simple_net}
    ```
    """)
    return SimpleCNN, simple_net


@app.cell
def _(mo, nn, np, optim, torch, tqdm):
    def simple_train(net, trainloader, epochs):
        # Initialize the network, loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.01)

        # Training the network and recording the loss
        num_epochs = epochs
        train_losses = []

        for epoch in range(num_epochs):
            running_loss = 0.0
            pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f'Epoch {epoch + 1}')
            for i, data in pbar:
                inputs, labels = data
                labels = torch.tensor(np.eye(10)[labels])  
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:
                    pbar.set_postfix({'Loss': running_loss / 100})
                    train_losses.append(running_loss / 100)
                    running_loss = 0.0
        print('Finished Training')

        return train_losses

    mo.md(text="""
    ### Step-3 设置训练函数

    这里我们定义一个训练函数，包含了训练的主要逻辑
    """)
    return (simple_train,)


@app.cell
def _(calculate_accuracy, mo, np, torch):
    def simple_test(net, testloader):
        # Testing the network
        testing_acc_phy = []

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                labels = torch.tensor(np.eye(10)[labels])

                outputs = net(images)
                correct_phy = calculate_accuracy(outputs, labels)
                testing_acc_phy.append(correct_phy)

        current_test_result = np.mean(testing_acc_phy)

        print( f'Phy accuracy of the network on 10000 test images: {100 * current_test_result:.3f}%')

    mo.md(text="""
    ### Step-4 设置测试函数

    测试函数用来对已经训练好的模型，观察它对没有见过的数据的表现如何。
    """)
    return (simple_test,)


@app.cell
def _(
    mo,
    plt,
    simple_net,
    simple_test,
    simple_test_button,
    simple_train,
    simple_train_button,
    simple_train_epoch_input,
    testloader,
    trainloader,
):
    if simple_train_button.value:
        _train_losses = simple_train(net=simple_net, trainloader=trainloader, epochs=simple_train_epoch_input.value)
        plt.plot(_train_losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()
        mo.md(text="训练完成，loss曲线如下所示")

    if simple_test_button.value:
        simple_test(net=simple_net, testloader=testloader)
        mo.md(text="测试完成，测试结果如下所示")

    mo.vstack([
        mo.md(text="""
        ### Step-5 开始训练/测试

        训练/测试常规的卷积神经网络识别MNIST。
        """),
        simple_train_epoch_input,
        mo.hstack([
            simple_train_button,
            simple_test_button        
        ])
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(text=f"""
    到此我们完成了实践二的第一部分内容，下面是一些辅助函数，不需要处理。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Show buttons here for user to start train/test process
    simple_train_epoch_input = mo.ui.number(label="Epochs", value=3, start=1, stop=10, step=1)
    simple_train_button = mo.ui.run_button(label="Start Train!")
    simple_test_button = mo.ui.run_button(label="Start Test!")
    return simple_test_button, simple_train_button, simple_train_epoch_input


@app.cell(hide_code=True)
def _(batchsize, mo, np, plt, torch, trainloader):
    # 可视化一些 MNIST 数据样本
    def visualize_mnist_samples():
        # 获取一批数据
        dataiter = iter(trainloader)
        images, labels = next(dataiter)

        # 创建一个图形来显示图像
        fig = plt.figure(figsize=(15, 6))

        # 显示一批数据中的前16张图像
        num_samples_to_show = min(16, len(images))

        for i in range(num_samples_to_show):
            ax = fig.add_subplot(2, 8, i+1)
            # 将图像从 [-1, 1] 转换回 [0, 1] 范围
            img = (images[i].squeeze().numpy() * 0.5 + 0.5)
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Label: {labels[i].item()}')
            ax.axis('off')

        plt.tight_layout()
        return fig

    # 可视化数据分布
    def visualize_mnist_distribution():
        # 获取所有训练集标签
        all_labels = []
        for _, labels in trainloader:
            all_labels.append(labels)
        all_labels = torch.cat(all_labels).numpy()

        # 计算每个类别的样本数量
        unique_labels, counts = np.unique(all_labels, return_counts=True)

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 绘制条形图
        ax1.bar(unique_labels, counts)
        ax1.set_title('MNIST Dataset Class Distribution')
        ax1.set_xlabel('Digit Class')
        ax1.set_ylabel('Sample Count')
        ax1.set_xticks(unique_labels)

        # 绘制饼图
        ax2.pie(counts, labels=unique_labels, autopct='%1.1f%%')
        ax2.set_title('MNIST Dataset Class Proportion')

        plt.tight_layout()
        return fig

    # 可视化一个批次的平均图像
    def visualize_batch_statistics():
        dataiter = iter(trainloader)
        images, labels = next(dataiter)

        # 计算批次的平均图像和标准差图像
        mean_img = torch.mean(images, dim=0).squeeze().numpy() * 0.5 + 0.5
        std_img = torch.std(images, dim=0).squeeze().numpy() * 0.5

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 显示平均图像
        im1 = ax1.imshow(mean_img, cmap='viridis')
        ax1.set_title(f'Batch Mean Image (batch size={batchsize})')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1)

        # 显示标准差图像
        im2 = ax2.imshow(std_img, cmap='plasma')
        ax2.set_title(f'Batch Standard Deviation Image (batch size={batchsize})')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        return fig

    # 组织所有可视化为一个选项卡界面
    def mnist_visualization_dashboard():
        return mo.ui.tabs({
            "Sample Preview": visualize_mnist_samples(),
            "Data Distribution": visualize_mnist_distribution(),
            "Batch Statistics": visualize_batch_statistics(),
        })
    return (
        mnist_visualization_dashboard,
        visualize_batch_statistics,
        visualize_mnist_distribution,
        visualize_mnist_samples,
    )


@app.cell(hide_code=True)
def calculate_accuracy():
    def calculate_accuracy(predictions, true_labels):
        """
        计算模型预测的准确率

        参数:
        predictions: 模型输出的预测结果，包含每个类别的得分或概率
        true_labels: 真实标签的one-hot编码或类别索引

        返回:
        float: 预测准确率，范围[0, 1]
        """
        correct_count = (predictions.argmax(dim=-1) == true_labels.argmax(dim=-1)).sum().item()
        accuracy = correct_count / predictions.size(0)
        return accuracy
    return (calculate_accuracy,)


@app.cell(hide_code=True)
def _(mo):
    batchsize_select = mo.ui.dropdown(label="选择`Batch`大小：", options=[8, 16, 32, 64], value=32)
    return (batchsize_select,)


@app.cell(column=4, hide_code=True)
def _(mo, practice_2_task_2, practice_2_title):
    mo.md(text=practice_2_title + practice_2_task_2)
    return


@app.cell(hide_code=True)
def _(batchsize_select, mo):
    mo.md(text=f"""
    ### Step-1 数据集

    这个数据集我们在之前常规的神经网络训练的时候就已经设置好了，这里不再重复操作。

    数据集batch大小: {batchsize_select.value}
    """)

    # batchsize = 32

    # trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
    # testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    return


@app.cell
def _(IMAGE_SIZE, crop_region, crop_size_input, display_cropped_images, mo):
    mo.md(text=f"""
    ### Step-1.5 设置裁切策略

    选择从中心/四个角裁切多大的图像。

    {mo.vstack([
        mo.hstack([crop_region, crop_size_input]),
        display_cropped_images(image_size=IMAGE_SIZE)
    ])}
    """)
    return


@app.cell
def _(IMAGE_SIZE, crop_size_input, mo, nn, torch):
    # Define the Complex neural network
    class ComplexLinear(nn.Module):
        def __init__(self, in_features, out_features):
            super(ComplexLinear, self).__init__()
            # 为方便，省略偏置项
            self.real_weight = nn.Parameter(torch.randn(in_features, out_features))
            self.imag_weight = nn.Parameter(torch.randn(in_features, out_features))

        def forward(self, real_input, imag_input):
            real_output = torch.matmul(real_input, self.real_weight) - torch.matmul(imag_input, self.imag_weight) 
            imag_output = torch.matmul(real_input, self.imag_weight) + torch.matmul(imag_input, self.real_weight) 
            return real_output, imag_output

    class ComplexReLU(nn.Module):
        def forward(self, real_input, imag_input):
            magnitude = torch.sqrt(real_input**2 + imag_input**2)
            phase = torch.atan2(imag_input, real_input)
            relu_magnitude = torch.relu(magnitude)
            return relu_magnitude * torch.cos(phase), relu_magnitude * torch.sin(phase)

    class ComplexCNN(nn.Module):
        def __init__(self, image_size, feature_map_num=16):
            super(ComplexCNN, self).__init__()
            self.image_size = image_size
            self.feature_map_num = feature_map_num
            self.conv1_real = nn.Conv2d(1, feature_map_num, kernel_size=3, stride=1, padding=1)
            self.conv1_imag = nn.Conv2d(1, feature_map_num, kernel_size=3, stride=1, padding=1)
            self.fc1 = ComplexLinear(feature_map_num * image_size * image_size, 10)
            self.relu = ComplexReLU()

        def forward(self, complex_input):
            real_input, imag_input = torch.real(complex_input), torch.imag(complex_input)

            real_output = self.conv1_real(real_input) - self.conv1_imag(imag_input)
            imag_output = self.conv1_real(imag_input) + self.conv1_imag(real_input)
            real_output, imag_output = self.relu(real_output, imag_output)

            real_output = real_output.view(-1, self.feature_map_num * self.image_size ** 2)
            imag_output = imag_output.view(-1, self.feature_map_num * self.image_size ** 2)
            real_output, imag_output = self.fc1(real_output, imag_output)

            magnitude_output = torch.sqrt(real_output**2 + imag_output**2)
            return magnitude_output


    crop_size = crop_size_input.value

    image_size = IMAGE_SIZE
    crop_center = image_size // 2

    complex_net = ComplexCNN(image_size=2*crop_size)
    mo.md(text=f"""
    ### Step-2 定义网络

    定义一个复数卷积神经网络。网络的结构如下:
    ```plaintext
    {complex_net}
    ```
    """)
    return (
        ComplexCNN,
        ComplexLinear,
        ComplexReLU,
        complex_net,
        crop_center,
        crop_size,
        image_size,
    )


@app.cell
def _(
    crop_fourier_domain,
    image_size,
    mo,
    nn,
    norm_inputs,
    np,
    optim,
    torch,
    tqdm,
):
    def complex_train(net, trainloader, epochs):
        # Initialize the network, loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.01)

        # Training the network and recording the loss
        num_epochs = epochs
        train_losses = []

        for epoch in range(num_epochs):
            running_loss = 0.0
            pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f'Epoch {epoch + 1}')
            for i, data in pbar:
                inputs, labels = data
                labels = torch.tensor(np.eye(10)[labels])

                inputs = np.fft.fftshift(np.fft.fft2(inputs.numpy()), axes=(2, 3))
                inputs = norm_inputs(inputs.reshape((-1, image_size**2))).astype(np.complex64)
                inputs = torch.from_numpy(inputs).view(-1, 1, image_size, image_size)

                inputs = crop_fourier_domain(inputs, image_size)

                # inputs = inputs[:, :, crop_center-crop_size: crop_center+crop_size, 
                                # crop_center-crop_size: crop_center+crop_size]

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:
                    pbar.set_postfix({'Loss': running_loss / 100})
                    train_losses.append(running_loss / 100)
                    running_loss = 0.0

        print('Finished Training')

        return train_losses

    mo.md(text="""
    ### Step-3 设置训练函数

    这里我们定义一个训练函数，包含了训练的主要逻辑
    """)
    return (complex_train,)


@app.cell
def _(
    calculate_accuracy,
    crop_center,
    crop_size,
    image_size,
    mo,
    norm_inputs,
    np,
    torch,
):
    def complex_test(net, testloader):
        # Testing the network
        testing_acc_phy = []

        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                labels = torch.tensor(np.eye(10)[labels])

                inputs = np.fft.fftshift(np.fft.fft2(inputs.numpy()), axes=(2, 3))
                inputs = norm_inputs(inputs.reshape((-1, image_size**2))).astype(np.complex64)
                inputs = torch.from_numpy(inputs).view(-1, 1, image_size, image_size)
                inputs = inputs[:, :, crop_center-crop_size: crop_center+crop_size, 
                                crop_center-crop_size: crop_center+crop_size]

                outputs = net(inputs)
                correct_phy = calculate_accuracy(outputs, labels)
                testing_acc_phy.append(correct_phy)

        current_test_result = np.mean(testing_acc_phy)

        # print(f'Complex network accuracy on 10000 test images: {100 * current_test_result:.3f}%')

        return current_test_result

    mo.md(text="""
    ### Step-4 设置测试函数

    测试函数用来对已经训练好的模型，观察它对没有见过的数据的表现如何。
    """)
    return (complex_test,)


@app.cell
def _(
    complex_net,
    complex_net_results,
    complex_test,
    complex_test_button,
    complex_train,
    complex_train_button,
    complex_train_epoch_input,
    crop_region,
    crop_size_input,
    display_complex_net_result,
    get_run_id,
    mo,
    plt,
    set_run_id,
    testloader,
    trainloader,
):
    if complex_train_button.value:
        _epochs = complex_train_epoch_input.value
        _train_losses = complex_train(net=complex_net, trainloader=trainloader, epochs=_epochs)
        plt.plot(_train_losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()
        current_run_id = get_run_id()

        complex_net_results[current_run_id] = {
            "Epoch": _epochs,
            "Crop Region": crop_region.value,
            "Crop Size": crop_size_input.value,
            "Train Loss": f"{_train_losses[-1]:.4f}",
            "Test Acc": "Not tested",
        }


    if complex_test_button.value:
        _complex_test_acc = complex_test(net=complex_net, testloader=testloader)

        current_run_id = get_run_id()
        complex_net_results[current_run_id]["Test Acc"] = _complex_test_acc

        # 当用户完成测试上个训练好的模型后，让id+1，准备开始训练下一个模型
        set_run_id(lambda run_id: run_id + 1)

    mo.vstack([
        mo.md(text="""
        ### Step-5 开始训练/测试

        训练/测试复数卷积神经网络识别MNIST。
        """),
        mo.hstack([
            mo.vstack([
                mo.md("训练参数设置:"),
                complex_train_epoch_input,
                mo.md("裁剪设置:"),
                mo.hstack([crop_region, crop_size_input])
            ]),
            mo.vstack([
                mo.md("操作:"),
                mo.hstack([
                    complex_train_button,
                    complex_test_button
                ])
            ])
        ], align="end"),
        display_complex_net_result(complex_net_results)
    ])
    return (current_run_id,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(text=f"""
    到此我们完成了所有的实践二内容，下面是一些辅助函数，不需要处理
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    complex_net_results = {}
    get_run_id, set_run_id = mo.state(0)

    def display_complex_net_result(data: dict):
        return mo.ui.table(data=list(data.values()), label="训练/测试结果记录")
    return (
        complex_net_results,
        display_complex_net_result,
        get_run_id,
        set_run_id,
    )


@app.cell(hide_code=True)
def _(np):
    def norm_inputs(inputs, feature_axis=1):
        if feature_axis == 1:
            n_features, n_examples = inputs.shape
        elif feature_axis == 0:
            n_examples, n_features = inputs.shape

        for i in range(n_features):
            l1_norm = np.mean(np.abs(inputs[i, :]))
            inputs[i, :] /= l1_norm
        return inputs
    return (norm_inputs,)


@app.cell(hide_code=True)
def _(crop_region, crop_size_input):
    def crop_fourier_domain(inputs, image_size):
        """
        Crop the Fourier domain based on user selected region and size

        Args:
            inputs: The input tensor in Fourier domain
            image_size: Size of the original image

        Returns:
            Cropped tensor
        """

        region = crop_region.value
        crop_size = crop_size_input.value

        # Determine crop center based on selected region
        if region == "Center":
            crop_center_x = image_size // 2
            crop_center_y = image_size // 2
        elif region == "Top Left":
            crop_center_x = crop_size
            crop_center_y = crop_size
        elif region == "Top Right":
            crop_center_x = image_size - crop_size
            crop_center_y = crop_size
        elif region == "Bottom Left":
            crop_center_x = crop_size
            crop_center_y = image_size - crop_size
        elif region == "Bottom Right":
            crop_center_x = image_size - crop_size
            crop_center_y = image_size - crop_size

        # Crop the input tensor
        cropped_inputs = inputs[:, :, 
                               crop_center_y-crop_size:crop_center_y+crop_size, 
                               crop_center_x-crop_size:crop_center_x+crop_size]

        return cropped_inputs
    return (crop_fourier_domain,)


@app.cell(hide_code=True)
def _(crop_region, crop_size_input, mo, np, plt):
    def display_cropped_images(image_size):
        crop_size = crop_size_input.value
        region = crop_region.value

        # Create a white image inside the function
        white_image = np.ones((image_size, image_size))

        # Determine crop center based on selected region
        if region == "Center":
            crop_center_x = image_size // 2
            crop_center_y = image_size // 2
        elif region == "Top Left":
            crop_center_x = crop_size
            crop_center_y = crop_size
        elif region == "Top Right":
            crop_center_x = image_size - crop_size
            crop_center_y = crop_size
        elif region == "Bottom Left":
            crop_center_x = crop_size
            crop_center_y = image_size - crop_size
        elif region == "Bottom Right":
            crop_center_x = image_size - crop_size
            crop_center_y = image_size - crop_size

        # Crop the original image
        cropped_spatial = white_image.copy()
        # Mark the cropped area in gray
        cropped_spatial[crop_center_y-crop_size:crop_center_y+crop_size, 
                       crop_center_x-crop_size:crop_center_x+crop_size] = 0.5

        # Calculate Fourier transform
        fft_image = np.fft.fft2(white_image)
        fft_shifted = np.fft.fftshift(fft_image)

        # Create a mask image representing the cropped area
        mask = np.zeros_like(white_image)
        mask[crop_center_y-crop_size:crop_center_y+crop_size, 
             crop_center_x-crop_size:crop_center_x+crop_size] = 1

        # Display images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image and cropped area
        axes[0].imshow(cropped_spatial, cmap='gray')
        axes[0].set_title('Spatial Domain Image and Crop Area')
        axes[0].axis('on')

        # Magnitude spectrum of Fourier transform
        magnitude_spectrum = np.abs(fft_shifted)
        axes[1].imshow(np.log1p(magnitude_spectrum), cmap='viridis')
        axes[1].set_title('Fourier Domain (Magnitude Spectrum, Log Scale)')
        axes[1].axis('on')

        # Cropped area in Fourier domain
        axes[2].imshow(np.log1p(magnitude_spectrum), cmap='viridis')
        axes[2].contour(mask, colors='r', linewidths=2)
        axes[2].set_title('Crop Area in Fourier Domain')
        axes[2].axis('on')

        plt.tight_layout()

        # Return crop parameter information
        crop_info = f"""
        ### Crop Parameters:
        - Crop Region: {region}
        - Crop Size: {crop_size}x{crop_size}
        - Crop Center Coordinates: ({crop_center_x}, {crop_center_y})
        - Resulting Image Size: {2*crop_size}x{2*crop_size}
        """

        return mo.vstack([
            mo.md(crop_info),
            mo.as_html(fig)
        ])
    return (display_cropped_images,)


@app.cell(hide_code=True)
def _(IMAGE_SIZE, mo):
    crop_region = mo.ui.dropdown(
        label="Crop Region",
        options=["Center", "Corner", "Top Left", "Top Right", "Bottom Left", "Bottom Right"],
        value="Center"
    )

    crop_size_input = mo.ui.number(label="Crop Size", start=1, stop=IMAGE_SIZE // 2, step=1, value=2)
    return crop_region, crop_size_input


@app.cell(hide_code=True)
def _(mo):
    # Show buttons here for user to start complex net train/test process
    complex_train_epoch_input = mo.ui.number(label="Epochs", value=3, start=1, stop=10, step=1)
    complex_train_button = mo.ui.run_button(label="Start Train!")
    complex_test_button = mo.ui.run_button(label="Start Test!")
    return complex_test_button, complex_train_button, complex_train_epoch_input


if __name__ == "__main__":
    app.run()
