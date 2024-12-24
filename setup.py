from setuptools import find_packages,setup
from typing import List

Hyper_e_dot = '-e .'

def get_requirements(file_path:str)-> List[str]:
    "This function will return the list of requrement"
    reqirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        reqirements = [reqirements.replace('\n','') for req in reqirements]
        if Hyper_e_dot in reqirements:
            reqirements.remove(Hyper_e_dot)
    return reqirements        


setup(
    name="mlproject",
    version='0.0.1',
    author='Prathamesh',
    author_email = 'prathameshbaviskar817@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirement.txt')

)