"""ocamcamera library for undistortion of Davide Scaramuzza's OcamCalib camera model

"""

from os import path

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(

    name='ocamcamera',  # Required
    version='0.0.2',  # Required
    description="Davide Scaramuzza's OcamCalib camera model undistortion library",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/matsuren/ocamcalib_undistort',  # Optional
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Researcher',
        'Topic :: Software Development :: Computer vision',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    packages=find_packages(),  # Required
    package_data={  # Optional
        'ocamcamera': ['img0.jpg', 'calib_results_0.txt'],
    },

    python_requires='>=3.6, <4',

    install_requires=['numpy', 'opencv-python'],

    project_urls={  # Optional
        'Bug Reports': 'https://github.com/matsuren/ocamcalib_undistort/issues',
        'Source': 'https://github.com/matsuren/ocamcalib_undistort/',
    },
)
