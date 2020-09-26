
import setuptools
from distutils.core import setup, Extension
import numpy as np
#sudo apt install libgsl-dev
setup(name='solve', version='1.0',   ext_modules=[Extension('c_solve', ["c_solve.c"], include_dirs=[np.get_include()], libraries=['m', 'gsl', 'gslcblas' ])], 
        author="Thomas Funck",
        author_email="tffunck@gmail.com",
        description="solve cubic polynomial",
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
