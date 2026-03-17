from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# =============================================================================
# PyTorch C++/CUDA 扩展编译脚本 (PyTorch C++/CUDA Extension Build Script)
# 
# 该脚本用于将 C++ 和 CUDA 源代码编译为 Python 可以直接导入的动态链接库。
# 编译后，在 Python 中可以通过 `import quadsim_cuda` 来调用底层的高性能物理和渲染函数。
# 
# 编译命令: python setup.py install
# =============================================================================

setup(
    name='quadsim_cuda', # 扩展模块的名称 (Name of the extension module)
    ext_modules=[
        CUDAExtension('quadsim_cuda', [
            'quadsim.cpp',        # C++ 接口绑定文件 (C++ interface binding file)
            'quadsim_kernel.cu',  # CUDA 渲染和碰撞检测内核 (CUDA rendering and collision kernels)
            'camera_fused_kernel.cu', # CUDA 融合相机 ISP kernel
            'dynamics_kernel.cu', # CUDA 物理动力学内核 (CUDA physics dynamics kernels)
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension # 使用 PyTorch 提供的 BuildExtension 来处理编译过程
    })
