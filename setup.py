from setuptools import find_packages, setup


HYPEN_E_DOT = '-e .'

def get_requirements(file_path):
    '''
    input : file path of requirements
    output: list of elements in requirements text file
    '''
    requirements = []
    with open(file_path) as file:
        requirements=file.readlines()
        requirements=[req.replace("\n"," ") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name = 'MedScraper',
    version = '0.0.1',
    author= 'Anduri Roshan',
    author_email='roshanandhuri@gmail.com',
    packages= find_packages(),
    install_required = get_requirements('requirements.txt')
)