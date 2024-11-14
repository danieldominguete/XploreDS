from setuptools import setup

with open("README.md", "r") as fh:
    readme = fh.read()

setup(
    name="xploreds",
    version=__version__,
    url="https://github.com/danieldominguete/XploreDS",
    license="MIT License",
    author="Daniel Dominguete",
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email="daniel.dominguete@gmail.com",
    keywords="Data Science",
    description="Easy-to-use package of the most relevant Data Science techniques.",
    packages=["xploreds"],
    install_requires=[
        "numpy==2.1.1",
        "pandas==2.2.3",
        "scikit_learn==1.5.2",
        "pyarrow==17.0.0",
        "statsmodels==0.14.4",
        "pydantic==2.9.2",
        "plotly==5.24.1",
        "kaleido==0.2.1",
        "python-dotenv",
        "lxml",
        "psutil",
    ],
    python_requires=">=3.10",
)
