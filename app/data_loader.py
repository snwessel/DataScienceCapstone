import csv
import pandas as pd
import numpy as np
from sodapy import Socrata

# Utils for loading data

class DataLoader:

  def get_states():
    """Load state abbreviations and names from file"""
    # states will be stored in a dict 
    # with the name as the key and the abbreviation as the value
    states_dict = {}
    with open('data/states.tsv', newline='') as states_file:
      tsv_reader = csv.reader(states_file, delimiter='\t')
      for row in tsv_reader:
        states_dict[row[0]] = row[1]
    return states_dict

  def get_daily_cases_df(state_abbrev):
    """Load daily case counts from the CDC API, return a pandas dataframe."""
    # Query the CDC API
    client = Socrata("data.cdc.gov", None)
    results = client.get("9mfq-cb36", state=state_abbrev)
    results_df = pd.DataFrame.from_records(results).sort_values(by=["created_at"])
    return results_df[["created_at", "new_case"]]


  def get_daily_cases_dict(state_abbrev):
    """Load daily case counts from the CDC API, return a dictionary which can be passed into JS"""
    results_df = DataLoader.get_daily_cases_df(state_abbrev)
    # return the new cases by date in a javascript-friendly format
    cases_by_date_dict = {
      "date": results_df["created_at"].tolist(),
      "cases": results_df["new_case"].tolist()
    }
    return cases_by_date_dict

  def get_daily_vaccinations_df(state_abbrev):
    """Load daily vaccination counts from CSV, return a pandas dataframe. CSV contains data up to 3/6/2021."""
    vax_df = pd.read_csv('data/us_state_vaccinations.csv')

    # data only includes full state name not abbreviations, so adding abbreviation column based on states_dict
    states_dict = DataLoader.get_states()
    vax_df["abbrev"] = vax_df["location"].map(states_dict)

    state_vax_df = vax_df[vax_df["abbrev"] == state_abbrev]
    daily_state_vax_df = state_vax_df[["date", "location", "abbrev", "daily_vaccinations"]]
    return daily_state_vax_df

  def get_daily_vaccinations_dict(state_abbrev):
    """Load daily vaccination counts from the CDC CSV, return a dictionary which can be passed into JS"""
    results_df = DataLoader.get_daily_vaccinations_df(state_abbrev)
    # return the new vaccinations by date in a javascript-friendly format
    vaccinations_by_date_dict = {
      "date": results_df["date"].tolist(),
      "vaccinations": results_df["daily_vaccinations"].tolist()
    }
    return vaccinations_by_date_dict

  def get_national_cases_df():
    """Load national daily case counts from the CDC API, return a pandas dataframe."""
    # Query the CDC API
    client = Socrata("data.cdc.gov", None)
    results = client.get("9mfq-cb36")
    results_df = pd.DataFrame.from_records(results).sort_values(by=["created_at"])

    # TODO: need to determine what to do about NY (New York State) vs NYC (New York City) as 2 separate entities
    #   filter results to only include 50 states (ALSO check if we should just be summarizing over everything including territories (i think no?))
    #   should DC be included? currently I have it but should that not count
    us_results_df = results_df[~results_df["state"].isin(["GU", "RMI", "MP", "PR", "VI", "PW", "FSM", "AS", "US"])]

    # aggregate by date
    us_results_df["new_case"] = us_results_df["new_case"].astype(float)
    # TODO: I think this aggregation isn't correct, some dates on the graph have 0 total cases listed which seems wrong (see JIRA ticket for image)
    agg_df = pd.DataFrame(us_results_df.groupby("created_at")["new_case"].agg(np.sum)).reset_index()

    return agg_df[["created_at", "new_case"]]

  def get_national_cases_dict():
    """Load national daily case counts from the CDC API, return a dictionary which can be passed into JS"""
    results_df = DataLoader.get_national_cases_df()
    # return the new cases by date in a javascript-friendly format
    cases_by_date_dict = {
      "date": results_df["created_at"].tolist(),
      "cases": results_df["new_case"].tolist()
    }
    return cases_by_date_dict

