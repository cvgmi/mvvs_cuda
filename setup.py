import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

os.system('make -j%d' % os.cpu_count())

# Python interface
setup(
    name='MVC',
    install_requires=['torch'],
    packages=['MVC'],
    package_dir={'MVC': './'},
    ext_modules=[
        CUDAExtension(
            name='MVC',
            include_dirs=['./'],
            sources=[
                'pybind/bind.cpp',
            ],
            libraries=['make_pytorch'],
            library_dirs=['objs'],
            #extra_compile_args=['-g']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
