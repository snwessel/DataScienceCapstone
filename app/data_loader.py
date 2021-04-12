import csv
from delphi_epidata import Epidata
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pickle
import requests
from sodapy import Socrata

# Utils for loading data

class DataLoader:

  ## Loading and Cleaning ##

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
    """Load daily case counts from the CDC API, clean data, return a pandas dataframe."""
    # Query the CDC API
    client = Socrata("data.cdc.gov", "qt5QX390BTNWFZ6O36g3oO6Fq")
    results = client.get("9mfq-cb36", state=state_abbrev)
    results_df = pd.DataFrame.from_records(results).sort_values(by=["submission_date"])

    # update the date formatting
    trimmed_df = results_df.replace(r'T\d{2}:\d{2}:\d{2}.\d{3}', '', regex=True)
    # remove duplicate case counts
    dup_idxs = np.where(trimmed_df["submission_date"].duplicated())
    trimmed_df = trimmed_df.reset_index()
    filtered_df = trimmed_df.drop(dup_idxs[0])
    # get a 7 day average
    filtered_df["new_case"] = filtered_df["new_case"].rolling(window=7).mean()
    filtered_df = filtered_df.dropna()
    return filtered_df[["submission_date", "new_case"]]


  def get_daily_cases_dict(daily_cases_df, state_abbrev):
    """Load daily case counts from the CDC API, return a dictionary which can be passed into JS"""
    # return the new cases by date in a javascript-friendly format
    cases_by_date_dict = {
      "date": daily_cases_df["submission_date"].tolist(),
      "cases": daily_cases_df["new_case"].tolist()
    }
    return cases_by_date_dict


  def get_daily_vaccinations_df(state_abbrev):
    """Load daily vaccination counts from CSV, return a pandas dataframe. CSV contains data up to 3/6/2021."""
    vax_df = pd.read_csv('data/us_state_vaccinations.csv')

    # data only includes full state name not abbreviations, so adding abbreviation column based on states_dict
    states_dict = DataLoader.get_states()
    states_dict["New York State"] = "NY" # address the fact that the csv calls NY "New York State"
    vax_df["abbrev"] = vax_df["location"].map(states_dict)

    state_vax_df = vax_df[vax_df["abbrev"] == state_abbrev]
    
    # get the daily total vaccinations per million
    vaccinations_per_million = state_vax_df["daily_vaccinations_per_million"].fillna(0).cumsum()
    state_vax_df.insert(0, "total_vaccinations_per_million", vaccinations_per_million)
    daily_state_vax_df = state_vax_df[["date", "location", "abbrev", "total_vaccinations_per_million"]]
    return daily_state_vax_df


  def get_daily_vaccinations_dict(daily_vaccinations_df, state_abbrev):
    """Load daily vaccination counts from the CDC CSV, return a dictionary which can be passed into JS"""
    # return the new vaccinations by date in a javascript-friendly format
    vaccinations_by_date_dict = {
      "date": daily_vaccinations_df["date"].tolist(),
      "vaccinations": daily_vaccinations_df["total_vaccinations_per_million"].tolist()
    }
    return vaccinations_by_date_dict


  def get_case_and_vax_df(case_df, vax_df):
    """Get a dataframe containing the daily case counts and vaccinations"""
    # merge the dataframes
    merged_df = pd.merge(case_df, vax_df, left_on="submission_date", right_on="date", how="left")
    merged_df.fillna(0, inplace=True)
    # pull out only the values we need
    return merged_df[["new_case", "total_vaccinations_per_million"]]


  def get_national_cases_df():
    """Load national daily case counts from the CDC API, return a pandas dataframe."""
    # Query the CDC API
    client = Socrata("data.cdc.gov", None)
    results = client.get("9mfq-cb36")
    results_df = pd.DataFrame.from_records(results).sort_values(by=["submission_date"])

    # TODO: need to determine what to do about NY (New York State) vs NYC (New York City) as 2 separate entities
    #   filter results to only include 50 states (ALSO check if we should just be summarizing over everything including territories (i think no?))
    #   should DC be included? currently I have it but should that not count
    us_results_df = results_df[~results_df["state"].isin(["GU", "RMI", "MP", "PR", "VI", "PW", "FSM", "AS", "US"])]

    # aggregate by date
    us_results_df["new_case"] = us_results_df["new_case"].astype(float)
    # TODO: I think this aggregation isn't correct, some dates on the graph have 0 total cases listed which seems wrong (see JIRA ticket for image)
    agg_df = pd.DataFrame(us_results_df.groupby("submission_date")["new_case"].agg(np.sum)).reset_index()

    return agg_df[["submission_date", "new_case"]]


  def get_national_cases_dict():
    """Load national daily case counts from the CDC API, return a dictionary which can be passed into JS"""
    results_df = DataLoader.get_national_cases_df()
    # return the new cases by date in a javascript-friendly format
    cases_by_date_dict = {
      "date": results_df["submission_date"].tolist(),
      "cases": results_df["new_case"].tolist()
    }
    return cases_by_date_dict

  def get_future_dates_list(last_date_str, num_days):
    future_dates = [] # initialize the array
    date_i = datetime.strptime(last_date_str, '%Y-%m-%d')
    for i in range(num_days):
        date_i += timedelta(days=1)
        date_str = date_i.strftime('%Y-%m-%d')
        future_dates.append(date_str) 
    return future_dates



  def get_assumed_vaccinations_dict(daily_total_vaccines_df, num_days, multiplier=1, past_days_referenced=14):
    """Generate a dictionary containing the predicted number of total vaccinations per day"""
    # use past vaccination information to make a prediction
    vax_per_million = daily_total_vaccines_df["total_vaccinations_per_million"]
    current_total_vaccines = vax_per_million.iloc[-1]
    start_total_vaccines = vax_per_million.iloc[-past_days_referenced]
    avg_per_day = ((current_total_vaccines - start_total_vaccines) / past_days_referenced) * multiplier
    predicted_daily_totals = pd.Series([avg_per_day]).repeat(num_days)
    predicted_daily_totals.iloc[0] += current_total_vaccines
    predicted_daily_totals = predicted_daily_totals.cumsum()

    # get the corresponding date strings for these values
    last_known_date = daily_total_vaccines_df["date"].iloc[-1]
    future_dates = DataLoader.get_future_dates_list(last_known_date, num_days)

    vaccinations_by_date_dict = {
      "date": future_dates,
      "vaccinations": predicted_daily_totals.tolist()
    }
    return vaccinations_by_date_dict 

  ## Processing for Training ##  

  def get_windowed_df(series, window_size):
    """ Convert a pandas series into a windowed dataframe. 
        Where the window size is k and the series size is n, the output should look like:
        [[x_0, x_1, ..., x_k-1]
         [x_1, x_2, ..., x_k],
         ...
         [          ..., x_n]
         """
    n_days = series.shape[0]
    # print("\tcreating windows from", n_days, "days of data.")
    n_windows = (n_days - window_size) + 1
    # initialize an array of the correct shape
    windowed_array = np.zeros((n_windows, window_size))
    # get the "windows" one at a time
    for i in range(n_windows):
        window = series[i:i+window_size]
        windowed_array[i] = window
    # convert to a dataframe
    windowed_df = pd.DataFrame(windowed_array)
    return windowed_df


  def get_state_windowed_training_data(window_size, state_abbrev, include_vaccinations=True, holdout_days=0):
    """ Get X and y values to use for training and testing a windowed model
        Where k is the window size, our outputs should look like the following:
        Our feature set X should look like:
        [[cases_0, cases_1, ... cases_k-2, vaccinations_k-2],
         [cases_1, cases_2, ... cases_k-1, vaccinations_k-1],
         [cases_2, cases_3, ... cases_k, vaccinations_k],
         ... ]
        Our y values should be:
        [cases_k-1, cases_k, cases_k+1, ...] 
        Note: The holdout_days arg specifies the number of most-recent days which should be excluded
              from training data (to use later in testing multi-day predictions)."""

    # load the case and vaccination data
    case_data = DataLoader.get_daily_cases_df(state_abbrev)
    vax_data = DataLoader.get_daily_vaccinations_df(state_abbrev)
    
    # get windowed case data
    df = DataLoader.get_case_and_vax_df(case_data, vax_data)
    if holdout_days > 0:
      df = df[:-holdout_days]
    X = DataLoader.get_windowed_df(df["new_case"], window_size)

    # make copy for y
    copy_X = X.copy()
    # pull out the final value in each window to use as our 'y' value
    y = copy_X[window_size-1]

    if include_vaccinations: 
      # add vaccination counts to the features
      vaccinations = df["total_vaccinations_per_million"]
      # we want the vaccination number to correspond to the last day of case counts in X
      vaccine_offset = window_size - 2
      offset_vaccinations = vaccinations[vaccine_offset:].reset_index(drop=True)
      X[window_size-1] = offset_vaccinations
    return X, y

  def get_windowed_training_data(window_size, include_vaccinations=True, holdout_days=0):
    """Get the combined windowed training data for all states."""
    # TODO: save state as boolean field
    # col_names = ["day" + str(x+1) for x in range(1, window_size)]
    # col_names.append("vaccinations_per_million")
    states = DataLoader.get_states()
    X = pd.DataFrame()
    y = pd.Series()
    for state_name, state_abbrev in states.items():
      # print("Getting data for", state_name)
      state_X, state_y = DataLoader.get_state_windowed_training_data(window_size, state_abbrev, include_vaccinations, holdout_days)
      X = X.append(state_X)
      y = y.append(state_y)
    return X, y

  def get_date_separated_testing_data(window_size, num_test_days):
    """Generate an X and y value for each state"""
    # load the case and vaccination data
    X, y = DataLoader.get_windowed_training_data(window_size)
    # generate X and y
    X_train = X[:-num_test_days] # all but last n elements
    y_train = y[:-num_test_days] # all but last n elements
    y_test = y[-num_test_days:] # last n elements
    return X_train, y_train, y_test


  ## Using the Trained Model ## 

  def get_predictions(case_df, vax_df, future_vaccinations_list, window_size, num_days):
    """Use the trained model to iteratively get predictions for future days."""
    # load the model
    model_bytes = pickle.load(open("data/trained_model.p", "rb"))
    model = pickle.loads(model_bytes)
    # assemble the set of features we'll feed in
    recent_cases = case_df["new_case"].tail(window_size-1).to_numpy().astype(int)
    current_vaccinations = vax_df["total_vaccinations_per_million"].iloc[-1]
    features = np.append(recent_cases, current_vaccinations)
    feature_matrix = np.atleast_2d(features)
    
    # iteratively get the predictions
    predictions = []
    for i in range(num_days):
      # get the prediction
      predicted = model.predict(feature_matrix).astype(int).item(0)
      predicted = max(0, predicted)
      predictions.append(predicted)
      # update the features
      past_cases = features[1:window_size-1]
      total_vaccinations = future_vaccinations_list[i]
      features = np.append(past_cases, [predicted, total_vaccinations])
      feature_matrix = np.atleast_2d(features)

    # convert to a dictionary 
    last_known_date = vax_df["date"].iloc[-1]
    future_dates = DataLoader.get_future_dates_list(last_known_date, num_days)
    predictions_dict = {
      "date": future_dates,
      "predictions": predictions
    }
    return predictions_dict


  ## Exploratory Datasets ##


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
    census_df = pd.DataFrame(columns=cols, data=results.json()[1:]).sort_values(by=['state_name'])

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
      "emergency_declaration": results_df["Emergency Declaration"].tolist()
    }
    return policy_dict

  # TODO: finish implementing
  def get_influenza_counts_df():
    """Load influenza counts from the CMU Delphi API, return a pandas dataframe"""
    # Retrieves national fluview data for each "epiweek" from 2020:
    # TODO: could try to compare this against COVID cases, either convert COVID cases to epiweeks or vice versa
    results = Epidata.fluview(["nat"], [Epidata.range(202001, 202053)])
    results_df = pd.DataFrame.from_records(results["epidata"]).sort_values(by=["epiweek"])
    results_df = results_df[["epiweek", "lag", "num_ili", "num_patients", "num_providers", "wili", "ili"]]
    return results_df

  def get_infuenza_counts_dict():
    """Load influenza counts from the CMU Delphi API, return a dictionary which can be passed into JS"""
    results_df = DataLoader.get_influenza_counts_df()
    # return the flu data in a javascript-friendly format
    influenza_dict = {
      "epiweek": results_df["epiweek"].tolist(),
      "lag": results_df["lag"].tolist(),
      "num_ili": results_df["num_ili"].tolist(),
      "num_patients": results_df["num_patients"].tolist(),
      "num_providers": results_df["num_providers"].tolist(),
      "wili": results_df["wili"].tolist(),
      "ili": results_df["ili"].tolist()
    }
    return influenza_dict