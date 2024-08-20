from setuptools import setup

setup(name='rocketcycles',
      version='1.0',
      description='Python package for analysis and sizing of staged combustion rocket engines',
      url='https://github.com/janstruzinski/RocketCycles.git',
      author='Jan Struzinski',
      license='MIT',
      packages=['rocketcycles'],
      install_requires=['rocketcea', 'scipy', 'numpy', 'pyfluids'],
      keywords='staged combustion liquid rocket engine cycle full flow oxidizer rich cea rocketcea',
      dependency_links=['https://github.com/ptgodart/nasaPoly.git'],
      zip_safe=False,
      )