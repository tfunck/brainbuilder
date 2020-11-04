import setuptools
from distutils.core import setup, Extension
import numpy as np

setup(name='c_upsample_mesh', version='1.0',   ext_modules=[Extension('c_upsample_mesh', ["c_upsample_mesh.c", "upsample_mesh.c"], include_dirs=[np.get_include()])], 
        author="Thomas Funck",
        author_email="tffunck@gmail.com",
        description="upsample obj mesh",
        #long_description=long_description,
        #long_description_content_type="text/markdown",
        url="https://github.com/tfunck/solve",
        packages=setuptools.find_packages(),
        classifiers=(
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            #"Operating System :: OS Independent",
            "Operating System :: POSIX :: Linux",
        ),
        
        
        )
