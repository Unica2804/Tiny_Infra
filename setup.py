import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = [
    os.path.join('source', 'swiglu_binding.cpp'),
    os.path.join('source', 'fused_swiglu.cu')
]

setup(
    name='custom_swiglu',
    packages=[],
    ext_modules=[
        CUDAExtension(
            name='custom_swiglu',
            sources=sources,
            # We add -arch=native so it optimizes for your specific RTX card's Tensor Cores
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-arch=native']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)