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