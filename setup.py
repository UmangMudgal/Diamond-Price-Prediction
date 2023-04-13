from setuptools import setup, find_packages
from typing import List

Hyphen_e_dot = '-e .'

def get_requirements(file_path:str)-> List[str]:
    """
    Description : This will return list of requirements.
    """

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements_list = [req.replace('\n', '') for req in requirements]

        if Hyphen_e_dot in requirements_list:
            requirements_list.remove(Hyphen_e_dot)

    return requirements_list


setup(
    name= 'Regression Project',
    version= '0.0.1',
    author='Umang Mudgal',
    author_email='mudgal0709@gmail.com',
    packages=find_packages(),
    requirements = get_requirements('requirements.txt')
)
