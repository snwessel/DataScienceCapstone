import csv
import requests
import pandas as pd
import numpy as np
from sodapy import Socrata
from delphi_epidata import Epidata

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

  def get_state_population_counts_df():
    """Load state population estimate counts from the Census Bureau API, return a pandas dataframe"""
    # TODO: Clean up request URL, probably don't need it in multiple parts
    host="https://api.census.gov/data"
    year="2019"
    dataset="pep/charagegroups"
    base_url="/".join([host, year, dataset])

    predicates = {}
    get_vars = ["NAME", "POP"]
    predicates["get"] = ",".join(get_vars)
    # Change to integer padded w/ space to get a specific state
    predicates["for"] = "state:*"
    results = requests.get(base_url, params=predicates)

    # TODO: fix error catching/how that relays into the frontend
    try:
      results.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # Not 200
        return "Error: " + str(e)

    cols = ["state_name", "state_pop", "state_abbrev"]
    census_df = pd.DataFrame(columns=cols, data=results.json()[1:])

    return census_df

  def get_state_population_counts_dict():
    """Load state population estimate counts from the Census Bureau API, return a dictionary which can be passed into JS"""
    results_df = DataLoader.get_state_population_counts_df()
    # return the new population coutns in a javascript-friendly format
    state_pop_dict = {
      "state": results_df["state_name"].tolist(),
      "population": results_df["state_pop"].tolist()
    }
    return state_pop_dict

  def get_social_distancing_df(state_abbrev):
    """Load most up to date social distancing policies from CSV, return a pandas dataframe. CSV contains data up to 3/9/2021."""
    # Columns:  ['Region', 'Status of Reopening', 'Stay at Home Order', 'Mandatory Quarantine for Travelers', 'Non-Essential Business Closures',
    #   'Large Gatherings Ban', 'Restaurant Limits', 'Bar Closures*', 'Statewide Face Mask Requirement', 'Emergency Declaration']
    # There is a row for the US or for a particular state
    social_dist_df = pd.read_csv("social_distancing_master_file.csv")
    social_dist_df.rename(columns={"Unnamed: 0": "Region"}, inplace=True)
    social_dist_df.drop(0, axis=0, inplace=True)

    # Mapping to state ID in case data is wanted for a state instead of the US
    states_dict = DataLoader.get_states()
    social_dist_df["Abbreviation"] = social_dist_df["Region"].map(states_dict)

    state_policy_df = social_dist_df[social_dist_df["Abbreviation"] == state_abbrev]
    return state_policy_df
  
  def get_social_distancing_dict(state_abbrev):
    """Load most up to date social distancing policies from CSV, return a dictionary which can be passed into JS"""
    # TODO: not sure what columns of importance we want to have JS accessible, so I kept them all...
    results_df = DataLoader.get_social_distancing_df(state_abbrev)
    # return the state policies in a javascript-friendly format
    policy_dict = {
      "reopening": results_df["Status of Reopening"].tolist(),
      "stay_at_home": results_df["Stay at Home Order"].tolist(),
      "mandatory_quarantine": results_df["Mandatory Quarantine for Travelers"].tolist(),
      "non_essential_closure": results_df["Non-Essential Business Closures"].tolist(),
      "large_gatherings": results_df["Large Gatherings Ban"].tolist(),
      "restaurant_lim": results_df["Restaurant Limits"].tolist(),
      "bar_closure": results_df["Bar Closures*"].tolist(),
      "mask_mandate": results_df["Statewide Face Mask Requirement"].tolist(),
      "emergency_declaration": results_df["Emergency Declaration"].tolist(),
    }
    return policy_dict

  # TODO: finish implementing
  def get_influenza_counts_df():
    """Load influenza counts from the CMU Delphi API, return a pandas dataframe"""
    # This retrieves national data, state data can be retrieved changing nat to state abbrev
    # TODO: figure out epiweeks??? and what equals which days??? if there are days?
    results = Epidata.fluview(regions="nat", epiweeks=202021)
    # Example results:
    # {'result': 1,
    #   'epidata': [{'release_date': '2021-03-05',
    #     'region': 'nat',
    #     'issue': 202108,
    #     'epiweek': 202021,
    #     'lag': 40,
    #     'num_ili': 11416,
    #     'num_patients': 1059901,
    #     'num_providers': 2963,
    #     'num_age_0': 900,
    #     'num_age_1': 2221,
    #     'num_age_2': None,
    #     'num_age_3': 4463,
    #     'num_age_4': 2208,
    #     'num_age_5': 1624,
    #     'wili': 1.00711,
    #     'ili': 1.07708}],
    #   'message': 'success'}
    retun None

  # TODO: implement
  def get_infuenza_counts_dict():
    pass