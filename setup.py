from setuptools import setup

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='markup-mna',
    version='1.0',
    description='markup-mna',
    packages=['markup-mna'],
    install_requires=reqs.strip().split('\n'),
    include_package_data=True,
)