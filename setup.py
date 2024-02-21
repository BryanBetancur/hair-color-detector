from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='hair_color_detector',
    version='0.5',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_data={'hair_color_detector': ['*.pth']},
    install_requires=[
        'setuptools',
        'Flask>=2.2.2',
        'numpy>=1.24.3',
        'opencv_contrib_python>=4.5.5.62',
        'Pillow>=9.4.0',
        'scikit_image>=0.19.3',
        'torch>=2.0.1',
        'torchvision>=0.15.2'

    ],
    # Metadatos
    author='Bryan Betancur',
    author_email='betan2@hotmail.com',
    description='Package used to get hair color or simularity between two hairs',
    url='https://github.com/BryanBetancur/hair-color-detector',
    license="MIT",
 
    # classifiers like program is suitable for python3, just leave as it is.
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)