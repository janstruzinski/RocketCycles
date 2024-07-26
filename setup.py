from setuptools import setup

setup(name='rocketcycles',
      version='0.1',
      description='Python package for analysis and sizing of staged combustion rocket engines',
      url='https://github.com/janstruzinski/RocketCycles.git',
      author='Jan Struzinski',
      license='MIT',
      packages=['rocketcycles'],
      install_requires=['rocketcea', 'scipy', 'numpy', 'nasaPoly', 'pyfluids'],
      keywords='staged combustion liquid rocket engine full flow',
      zip_safe=False)