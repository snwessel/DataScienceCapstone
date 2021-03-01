import csv
import pandas as pd
from sodapy import Socrata

# Class definitions for any python utils

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

  def get_daily_cases(state_abbrev):
    """Load daily case counts from the CDC API"""
    # Query the CDC API
    client = Socrata("data.cdc.gov", None)
    results = client.get("9mfq-cb36", state=state_abbrev)
    results_df = pd.DataFrame.from_records(results)
    # return the new cases by date in a javascript-friendly format
    cases_by_date_df = results_df[["new_case", "created_at"]]
    cases_by_date_df = cases_by_date_df.sort_values(by=['created_at'])
    return list(cases_by_date_df.T.to_dict().values())

  def get_daily_vaccinations(state_abbrev):
    """Load daily vaccination counts from CSV"""
    # TODO: implement
    pass
