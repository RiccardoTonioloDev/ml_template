from setuptools import setup, find_packages

setup(
    #TODO: configure with your own name
    name="ml_template",
    #TODO: configure with your own description
    description='A template repository to standardize boilerplate code for ML research.',
    long_description=open('README.md').read(),
    #TODO: configure with your own version
    version='0.1.0',
    url='https://github.com/riccardotoniolodev/ml_template',
    author='Riccardo Toniolo',
    author_email='ssctonioloriccardo@gmail.com',
    python_requires='>=3.11',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=open('requirements.txt').readlines()
)