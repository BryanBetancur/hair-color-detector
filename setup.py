from setuptools import setup, find_packages

setup(
    name='hair_color_detector',
    version='0.1',
    packages=find_packages(),
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
    description='Package used to get hair color or simularity beetween two hairs',
    url='https://github.com/BryanBetancur/hair-color-detector',
)