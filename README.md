# XploreDS

Easy-to-use package of the most relevant Data Science techniques.
 
# Getting Started

## Dependencies

You need Python 3.9 or later to use **XploreDS**. You can find it at [python.org](https://www.python.org/).

## Library installation

For just use the library XploreDS in your project:

```
pip install xploreds
```

## Cookbooks 

For use the boilerplates with examples of XploreDS applications:

Clone this repo to your local machine using:

```
git clone https://github.com/danieldominguete/XploreDS
```

and explore the `src/cookbook` for standalone scripts or execute the customized pipeline execution with 

```
python main_pipeline_execution.py -f pipeline_config/CONFIG_FILE.json
```

## Dataset 

This project uses the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) as the standard example to illustrate the features of the **XploreDS** package.

This dataset provides detailed information about orders placed on a large e-commerce platform in Brazil, including data on customers, products, payments, reviews, and deliveries. It is widely used in data science projects to demonstrate techniques for analysis, preprocessing, data mining, and predictive modeling.

Please download it and save all .csv files at `data/credit-g/raw/ecommerce/` folder.

Throughout the documentation and cookbooks, practical examples will use this dataset to show how to apply XploreDS tools to real-world data analysis problems.

# Library Modules

## Data Acquisition

Data acquisition is the process of gathering and measuring information on variables of interest in an established systematic fashion. 

In the context of data science, it involves collecting data from various sources to be used for analysis, modeling, and decision-making. 

The XploreDS library provides tools to facilitate the acquisition of datasets, ensuring that the data is readily available for subsequent processing and analysis.

Subjects and Tools:
 
- [Data Repositories Integration](notes/data_acquisition.md#public-datasets-integrations)

## Data Preprocessing

- Rename columns :hourglass_flowing_sand:
- Missing values :hourglass_flowing_sand:
- 
## Data Mining

[Data Analysis](notes/data_analysis.md)

## Modeling

## xAI

# Bug Reports

Bug reports can be submitted to the issue tracker:

https://github.com/danieldominguete/XploreDS/issues

# Changelog

All notable changes to this project will be documented [here](CHANGELOG.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).