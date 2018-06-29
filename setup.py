import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="textsim",
    version="0.0.1",
    author="Kamran Haider",
    author_email="kamranhaider.mb@gmail.com",
    description="A Python package to compare documents ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kamran-haider/TextSim",
    install_requires=[
        'pytest', 'scikit-learn', 'pandas', 'numpy', 'nltk'],
    packages=setuptools.find_packages()
)