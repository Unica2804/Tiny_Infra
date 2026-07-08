import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = [
    os.path.join('source', 'wmma_binding.cpp'),
    os.path.join('source', 'wmma.cu')
]

setup(
    name='custom_ops',
    ext_modules=[
        CUDAExtension(
            name='custom_ops',
            sources=sources,
            # We add -arch=native so it optimizes for your specific RTX card's Tensor Cores
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-arch=native']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)