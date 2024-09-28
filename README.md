# XploreDS

Repository of examples of Data Science cookbooks and library.

## Repository Organization

- data: folder of data files
- src: source codes with examples
- lib: library of encapsulted methods
- docs: most relevant technical documents
- guides: "how to" documentation of main tools
- output: folder for outputs produced by scripts
- static: static files of project 
- tests: test scripts  
  
## Setup Environment

1 - Install virtual environment package

`pip install virtualenv`

2 - Create the virtual environment

`virtualenv XploreDS`

3 - Activate the virtual environment

Mac OS/Linux: `source XploreDS/bin/activate`

Windows: `XploreDS\Scripts\activate`

4 - Install requirement packages

`pip install -r requirements`

5 - Create .env file and/or set environment variables

`PYTHON_WARNINGS="ignore"`

 

# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1]

### Added
- Script management functions: logging, working folder, time spent
- Project conception by [@danieldominguete](https://github.com/danieldominguete).