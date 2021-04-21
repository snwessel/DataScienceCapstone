# DataScienceCapstone

This project aims to explore data surrounding COVID cases and vaccinations in the United States. Our primary deliverable takes the form of a webapp, where users will be able to explore data visualizations and interact with our predictive model. 

[View our website here!](https://capstone-covid-predictions.herokuapp.com) 

_Created with love by Sarah Wessel and Olivia Petrillo._

## Local Setup Instructions
If you don't already have `virtualenv` installed, run: 
```
python -m pip install --user virtualenv
``` 

Next, install requirements:
```
python -m venv env
python -m pip install --user -r requirements.txt
```

Lastly, run the app! 
```
python run.py
```

## Updating Requirements
Generate a new requirements file by running the following (with your own file path):
```
pip install pipreqs
pipreqs \Users\...\DataScienceCapstone\app
```

## Data Sources
* Daily case counts and deaths per state from the CDC **(API)**
    * https://dev.socrata.com/foundry/data.cdc.gov/9mfq-cb36
* Daily vaccinations per state from Our World in Data **(CSV)**
    * https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/us_state_vaccinations.csv
* Temporal influenza (and other epidemic) data from Carnegie Mellon University Delphi research group **(API)**
    * https://cmu-delphi.github.io/delphi-epidata/api/fluview.html
* 2019 _Population Estimates: Estimates by Age Group, Sex, Race, and Hispanic Origin_ from the U.S. Census Bureau **(API)**
    * https://www.census.gov/data/developers/guidance/api-user-guide.Overview.html
* COVID-19 social distancing actions per state from KFF **(CSV)**
    * https://github.com/KFFData/COVID-19-Data/blob/kff_master/State%20Policy%20Actions/State%20Social%20Distancing%20Actions/Master%20File_Social%20Distancing.csv