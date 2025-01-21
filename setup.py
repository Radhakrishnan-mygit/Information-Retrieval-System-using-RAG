from setuptools import find_packages,setup
from typing import List

hypen_e="-e ."
def get_requirements(filepath:str)->List[str]:
   requirements=[]
   with open(filepath,'r') as file:
      requirements=file.readlines()
      requirements=[line.strip() for line in requirements if line.strip() and not line.startswith('#')]
      if hypen_e in requirements:
         requirements.remove(hypen_e)
   return requirements

setup(
   name="Generative ai project",
   version="0.0.1",
   author="radha",
   author_email="knownradha7@gmail.com",
   packages=find_packages(),
   install_requires=get_requirements("requirements.txt")
)