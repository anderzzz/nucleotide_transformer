from setuptools import setup

setup(
    name='Biosequences',
    url='https://github.com/anderzzz/nucleotide_transformer',
    author='Anders Ohrn',
    author_email='ohrn.anders@gmail.com',
    packages=['biosequences'],
    install_requires=['torch', 'pandas', 'transformers', 'datasets'],
    version='0.1',
    license='MIT',
    description='Transformer method for nucleotide sequence inputs',
    long_description=open('README.txt').read()
)