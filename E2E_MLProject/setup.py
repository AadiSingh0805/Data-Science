from setuptools import find_packages,setup

def get_req(file):
    requirements = []
    with open(file) as obj:
        requirements = obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
    
    return requirements

setup(
    name='E2EMLProject',
    version='0.0.1',
    author='Aadi',
    packages=find_packages(),
    install_requires=get_req('requirements.txt')
)