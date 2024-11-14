from setuptools import setup

with open("README.md", "r") as fh:
    readme = fh.read()

setup(
    name="xploreds",
    version="0.0.1",
    url="https://github.com/danieldominguete/XploreDS",
    license="MIT License",
    author="Daniel Dominguete",
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email="daniel.dominguete@gmail.com",
    keywords="Data Science",
    description="Easy-to-use package of the most relevant Data Science techniques.",
    packages=["xploreds"],
    install_requires=["numpy", "pandas"],
)
