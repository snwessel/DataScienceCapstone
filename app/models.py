import csv

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
    # TODO: implement
    pass

  def get_daily_vaccinations(state_abbrev):
    """Load daily vaccination counts from CSV"""
    # TODO: implement
    pass
