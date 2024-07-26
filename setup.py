from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.rst").read_text()

setup(name='rocketcycles',
      version='0.1',
      description='Python package for analysis and sizing of staged combustion rocket engines',
      url='https://github.com/janstruzinski/RocketCycles.git',
      author='Jan Struzinski',
      license='MIT',
      packages=['rocketcycles'],
      install_requires=['rocketcea', 'scipy', 'numpy', 'pyfluids'],
      keywords='staged combustion liquid rocket engine full flow',
      dependency_links=['https://github.com/ptgodart/nasaPoly.git'],
      zip_safe=False,
      long_description=long_description,
      long_description_content_type='text/x-rst'
      )