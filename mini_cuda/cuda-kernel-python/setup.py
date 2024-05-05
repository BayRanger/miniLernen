import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

os.system('make -j%d' % os.cpu_count())

setup(
    name='MultiPyTorchCUDAKernel',
    version='1.0.0',
    install_requires=['torch'],
    packages=['MultiPyTorchCUDAKernel'],
    package_dir={'MultiPyTorchCUDAKernel': './'},
    ext_modules=[
        CUDAExtension(
            name='DummMultiply',
            include_dirs=['./'],
            sources=[
                'pybind/bind.cpp',
            ],
            libraries=['multiply_pytorch'],
            library_dirs=['lib_multiply'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    author='Changxin',
    author_email='hello@bye.com',
    description='Custom PyTorch CUDA Extension for mutrix multi',
    keywords='PyTorch CUDA Extension',
    url='www.google.com',
    zip_safe=False,
)
