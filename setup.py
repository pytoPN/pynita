from setuptools import setup, find_packages

setup(name='pynita',
      version='0.1',
      description='Python version of NITA',
      url='http://github.com/fengly20/pynita',
      author='Leyang Feng',
      author_email='feng@american.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy>=1.14',
                        'scipy>=1.0',
                        'tqdm>=4.23',
                        'gdal>=2.1',
                        'pandas>=0.22',
                        'configobj>=5.0',
                        'matplotlib>=2.1',
                        'PyQt5>=5.9.2'],
      zip_safe=False)
