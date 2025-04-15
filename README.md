# ICE2606 Labs

本仓库是上海交通大学 ICE2606 课程的实验代码。课程学期是 2025 春季学期。

## Setup

本仓库主要利用[marimo](https://marimo.io/)，提供了一个交互式的 Python 环境，非常适合数据探索、实验和教学。`marimo` 的主要好处包括：

-   **交互性**：`marimo` 提供了一个 notebook 风格的界面，支持实时代码执行和可视化输出，方便快速迭代和调试。
-   **可重现性**：代码、数据和可视化结果可以轻松共享，确保实验结果的一致性。
-   **易用性**：与传统的 Jupyter Notebook 相比，`marimo` 提供了更现代化的界面和更流畅的用户体验。

### 环境设置

先克隆本仓库：

```bash
git clone https://github.com/ukeSJTU/ICE2606.git
```

为了运行本仓库中的代码，你需要设置一个 Python 虚拟环境。以下是两种方法，我们更推荐使用 `uv` 工具，因为它更快且更现代。

#### 方法 1：使用 `uv`（推荐）

`uv` 是一个现代的 Python 包和环境管理工具，速度比传统的 `venv` 和 `pip` 更快，且支持锁文件以确保依赖的一致性。如果你还没有安装 `uv`，可以按照[官方指南](https://github.com/astral-sh/uv)进行安装。

```bash
# 克隆仓库后，进入项目目录
cd ICE2606

# 使用 uv 创建并激活虚拟环境，同时安装依赖
uv sync
```

这将自动创建虚拟环境并安装 `pyproject.toml` 中定义的所有依赖项。

#### 方法 2：使用 `python -m venv`

如果你更喜欢使用 Python 内置的工具，可以按照以下步骤创建虚拟环境：

```bash
# 克隆仓库后，进入项目目录
cd ICE2606

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境（macOS/Linux）
source .venv/bin/activate
# 或者（Windows）
.venv\Scripts\activate

# 安装依赖
pip install -e .
```

激活环境后，你可以运行 `marimo` 或其他项目脚本。

### 运行 marimo

设置好环境后，你可以启动 `marimo`：

```bash
marimo run path/to/notebook.py
```

这将打开一个浏览器窗口，展示交互式 notebook 界面。

## Acknowledgments

[audio](./Lab2/demo/audio.wav) is a sound file used for demonstration purposes. It is downloaded from [2empower](https://2empower.com/great-audio-example/)
