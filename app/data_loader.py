import csv
import pickle
import requests
import numpy as np
import pandas as pd
from sodapy import Socrata
from delphi_epidata import Epidata
from datetime import datetime, timedelta

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


  def get_daily_cases_dict(daily_cases_df):
    """Load daily case counts from the CDC API, return a dictionary which can be passed into JS"""
    # return the new cases by date in a javascript-friendly format
    cases_by_date_dict = {
      "date": daily_cases_df["submission_date"].tolist(),
      "cases": daily_cases_df["new_case"].tolist()
    }
    return cases_by_date_dict


  def get_daily_vaccinations_df(state_abbrev):
    """Load daily vaccination counts from CSV, return a pandas dataframe."""
    vax_df = pd.read_csv('data/us_state_vaccinations.csv')

    # data only includes full state name not abbreviations, so adding abbreviation column based on states_dict
    states_dict = DataLoader.get_states()
    states_dict["New York State"] = "NY" # address the fact that the csv calls NY "New York State"
    vax_df["abbrev"] = vax_df["location"].map(states_dict)

    state_vax_df = vax_df[vax_df["abbrev"] == state_abbrev]
    
    # get the daily total vaccinations per million
    renamed_state_vax_df = state_vax_df.rename(columns={"people_fully_vaccinated_per_hundred": "percent_vaccinated"})
    filled_state_vax_df = renamed_state_vax_df.fillna(method="ffill") # fill missing vals with last known val
    daily_state_vax_df = filled_state_vax_df[["date", "abbrev", "percent_vaccinated"]]
    return daily_state_vax_df


  def get_daily_vaccinations_dict(daily_vaccinations_df, state_abbrev):
    """Load daily vaccination counts from the CDC CSV, return a dictionary which can be passed into JS"""
    # return the new vaccinations by date in a javascript-friendly format
    vaccinations_by_date_dict = {
      "date": daily_vaccinations_df["date"].tolist(),
      "vaccinations": daily_vaccinations_df["percent_vaccinated"].tolist()
    }
    return vaccinations_by_date_dict


  def get_case_and_vax_df(case_df, vax_df):
    """Get a dataframe containing the daily case counts and vaccinations"""
    # merge the dataframes
    merged_df = pd.merge(case_df, vax_df, left_on="submission_date", right_on="date", how="left")
    merged_df.fillna(0, inplace=True)
    # pull out only the values we need
    return merged_df[["new_case", "percent_vaccinated"]]


  def get_national_cases_df(date_bound=None):
    """Load national daily case counts from the CDC API, return a pandas dataframe. If specified, only return cases up to the date_bound."""
    # Query the CDC API
    client = Socrata("data.cdc.gov", "qt5QX390BTNWFZ6O36g3oO6Fq")
    results = client.get("9mfq-cb36", limit=50000)
    results_df = pd.DataFrame.from_records(results).sort_values(by=["submission_date"])
    
    # update the date formatting
    trimmed_df = results_df.replace(r'T\d{2}:\d{2}:\d{2}.\d{3}', '', regex=True).copy()

    # aggregate by date
    trimmed_df["new_case"] = trimmed_df["new_case"].astype(float)
    agg_df = pd.DataFrame(trimmed_df.groupby("submission_date")["new_case"].agg(np.sum)).reset_index()

    national_cases_df = agg_df.copy().reset_index()
    national_cases_df = national_cases_df[["submission_date", "new_case"]]

    if date_bound is not None:
        national_cases_df = national_cases_df[national_cases_df["submission_date"] < date_bound]

    # get a 7 day average
    national_cases_df["new_case"] = national_cases_df["new_case"].rolling(window=7).mean()
    national_cases_df = national_cases_df.dropna()

    return national_cases_df

  def get_national_cases_dict(national_cases_df):
    """Load national daily case counts from the CDC API, return a dictionary which can be passed into JS"""
    # return the new cases by date in a javascript-friendly format
    cases_by_date_dict = {
      "date": national_cases_df["submission_date"].tolist(),
      "cases": national_cases_df["new_case"].tolist()
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
    vax_per_million = daily_total_vaccines_df["percent_vaccinated"]
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
      vaccinations = df["percent_vaccinated"]
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
      if state_abbrev != "US":
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
    current_vaccinations = vax_df["percent_vaccinated"].iloc[-1]
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

    
  def get_future_case_bounds(predictions_dict, num_prev_days):
    # TODO: get these numbers for a larger number of days, then only index the ones we need
    mean_error = [ 31.1, 50.04, 73.06, 96.08, 125.56, 155.66, 178.68, 207.84, 224.58, 238.64,
      272.02, 266.2, 294.96, 290.06, 295.74, 304.4, 335.04, 342.08, 387.62, 391.18, 417.88] # observed mean error for the 21 predicted days
    predictions = np.array(predictions_dict["predictions"])
    upper = predictions+mean_error
    lower = predictions-mean_error
    return {
      "upper": upper.tolist(),
      "lower": lower.tolist()
    }

  # NOTE: I'm leaving this commented out because I'm not sure if we'll
  #  want to mention in our paper that we tried this and describe why it didn't work. 
  #
  # def get_future_case_bounds(daily_cases_df, num_to_predict, standard_devs):
  #   """Get the upper and lower limit of probable case counts by modeling the
  #      case counts as geometric brownian motion, and calculating the variance
  #      of that statistical model."""
  #   # Note: To avoid an overflow error in calculating e**a where a is around 60k, we need to make the numbers much smaller
  #   # calculate the average drift and variance of the observed data so far.
  #   case_counts = np.array(daily_cases_df["new_case"]) / 1000.0 # will multiply this back in at the end
  #   n = case_counts.shape[0]
  #   print("________________________")
  #   drift = (case_counts[-1] - case_counts[0]) / n # the average change in case counts change per day
  #   print("Drift:", drift)
  #   expected_past = pd.Series([drift]).repeat(n)
  #   expected_past.iloc[0] += case_counts[0]
  #   expected_past = expected_past.cumsum()
  #   print("Expected past case counts:", expected_past)
  #   variance = np.var(case_counts - expected_past)
  #   print("Variance:", variance)
  #   # calculate the expected future values
  #   expected_future = []
  #   future_variance = []
  #   for t in range(1, num_to_predict+1):
  #     # calculate the expected value at time t
  #     exponent = (drift + variance/2.0)*t
  #     expected_at_t = case_counts[-1] * np.exp(exponent)
  #     expected_future.append(expected_at_t)
  #     # calculate the variance at time t
  #     exponent = (drift + variance)*2*t
  #     print("variance exponent", exponent)
  #     multiplier = 1 - np.exp(-1*variance*t)
  #     variance_at_t = (case_counts[-1]**2) * np.exp(exponent) * multiplier
  #     future_variance.append(variance_at_t)
  #   # calculate upper and lower bounds 
  #   bound_range = standard_devs * np.sqrt(np.array(future_variance))
  #   upper_bound = (np.array(expected_future) + bound_range) * 1000
  #   lower_bound = (np.array(expected_future) - bound_range) * 1000
  #   return {
  #     "upper": upper_bound.tolist(),
  #     "lower": lower_bound.tolist()
  #   }


  ## Exploratory Datasets ##

  # NOTE: not fixed/re-implemented yet
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

  # NOTE: not fixed/re-implemented yet
  def get_state_population_counts_dict():
    """Load state population estimate counts from the Census Bureau API, return a dictionary which can be passed into JS"""
    results_df = DataLoader.get_state_population_counts_df()
    # return the new population coutns in a javascript-friendly format
    state_pop_dict = {
      "state": results_df["state_name"].tolist(),
      "population": results_df["state_pop"].tolist()
    }
    return state_pop_dict

  def get_state_policy_actions():
    """Return list of state policy actions to be loaded into a dropdown button."""
    social_dist_df = pd.read_csv("data/social_distancing_master_file.csv")
    social_dist_df.drop('Unnamed: 0', axis=1, inplace=True)
    
    return list(social_dist_df.columns)
  
  def get_state_policy_df(state_policy):
    """Load most up to date social distancing policies from CSV, return a pandas dataframe."""
    # Columns:  ['Region', 'Status of Reopening', 'Stay at Home Order', 'Mandatory Quarantine for Travelers', 'Non-Essential Business Closures',
    #   'Large Gatherings Ban', 'Restaurant Limits', 'Bar Closures*', 'Statewide Face Mask Requirement', 'Emergency Declaration']
    # There is a row for the US or for a particular state
    state_policy_df = pd.read_csv("data/social_distancing_master_file.csv")
    state_policy_df.rename(columns={"Unnamed: 0": "Region"}, inplace=True)
    state_policy_df.drop(0, axis=0, inplace=True)

    # Mapping to state ID in case data is wanted for a state instead of the US
    states_dict = DataLoader.get_states()
    state_policy_df["Abbreviation"] = state_policy_df["Region"].map(states_dict)

    # Clean up Large Gatherings Ban column to ensure text is consistent
    state_policy_df["Large Gatherings Ban"] = state_policy_df["Large Gatherings Ban"].str.replace('>\s', '>', regex=True)

    # Remove DC since DC isn't supported by the chloropleth version we are plotting
    state_policy_df = state_policy_df[state_policy_df["Abbreviation"] != "DC"]

    # Mapping policy type to index to create a factor so the values can be used for visualizations
    policy_nums = list(np.unique(state_policy_df[state_policy].values))
    
    # Moving 'All Gatherings Prohibited' to the front so it is in order of severity/restrictedness
    if state_policy == "Large Gatherings Ban":
      policy_nums.insert(0, policy_nums.pop(policy_nums.index("All Gatherings Prohibited")))

    policy_dict = {policy: idx for idx, policy in enumerate(policy_nums)}

    # Mapping dict to dataframe to add id column for policy type
    state_policy_df["Policy ID"] = state_policy_df[state_policy].map(policy_dict)
    return state_policy_df
  
  def get_state_policy_dict(state_policy_df, state_policy):
    """Load most up to date social distancing policies from CSV, return a dictionary which can be passed into JS"""    
    # Since we can't do categorical chloropleths easily, break dataframe up by "trace" (Policy ID)
    policy_ids = sorted(list(np.unique(state_policy_df["Policy ID"].values)))

    policy_dict = {"policy_name": state_policy}
    for policy_id in policy_ids:
        policy_df = state_policy_df[state_policy_df["Policy ID"] == policy_id]
        
        sub_policy_dict = {
            "state": policy_df["Region"].tolist(),
            "state_abbrev": policy_df["Abbreviation"].tolist(),
            "policy_info": policy_df[state_policy].tolist(),
            "policy_id": policy_df["Policy ID"].tolist(),
        }
        
        policy_dict[str(policy_id)] = sub_policy_dict

    # return the state policies in a javascript-friendly format by trace
    return policy_dict

  def get_approx_date_from_epiweek(epiweek):
    """Calculates approximate date based on epiweek. Epiweek formatted YYYYWW, with WW meaning week between 01-53."""
    year = int(str(epiweek)[:4])
    week = int(str(epiweek)[4:])

    if week == 53:
      days = (1 + (week - 1) * 7)
    else:
      days = week * 7
    
    new_date = datetime(year, 1, 1)
    new_date = new_date + timedelta(days=days - 1)

    # Format it as a string to match COVID date formatting
    return new_date.strftime("%Y-%m-%d")

  def get_influenza_counts_df():
    """Load influenza counts from the CMU Delphi API, return a pandas dataframe"""
    # Retrieves national fluview data for each "epiweek" from 2020:
    results = Epidata.fluview(["nat"], [Epidata.range(202001, 202053)])
    results_df = pd.DataFrame.from_records(results["epidata"]).sort_values(by=["epiweek"])
    results_df = results_df[["epiweek", "lag", "num_ili", "num_patients", "num_providers", "wili", "ili"]]

    # Convert epiweeks to approximate real date for graphing
    results_df["date"] = results_df["epiweek"].apply(DataLoader.get_approx_date_from_epiweek)
    return results_df

  def get_influenza_counts_dict(influenza_df):
    """Load influenza counts from the CMU Delphi API, return a dictionary which can be passed into JS"""
    # return the flu data in a javascript-friendly format
    influenza_dict = {
      "epiweek": influenza_df["epiweek"].tolist(),
      "date": influenza_df["date"].tolist(),
      "lag": influenza_df["lag"].tolist(),
      "num_ili": influenza_df["num_ili"].tolist(),
      "num_patients": influenza_df["num_patients"].tolist(),
      "num_providers": influenza_df["num_providers"].tolist(),
      "wili": influenza_df["wili"].tolist(),
      "ili": influenza_df["ili"].tolist()
    }
    return influenza_dict

  def get_total_vaccinations_per_hundred_df():
      """Load total vaccine doses administered per hundred for each state (from CSV), return a pandas dataframe."""
      vax_df = pd.read_csv('data/us_state_vaccinations.csv')

      # data only includes full state name not abbreviations, so adding abbreviation column based on states_dict
      states_dict = DataLoader.get_states()
      states_dict["New York State"] = "NY" # address the fact that the csv calls NY "New York State"
      vax_df["abbrev"] = vax_df["location"].map(states_dict)

      # filter by most recent date
      latest_date = max(list(vax_df["date"].values))
      latest_date_df = vax_df[vax_df["date"] == latest_date]   

      # only include 50 US states, DC also has to be excluded (for plotly chloropleth constraints)
      us_results_df = latest_date_df[~latest_date_df["abbrev"].isin(["DC", np.nan])]

      total_vax_national_df = us_results_df[["date", "location", "abbrev", "total_vaccinations_per_hundred"]]
      return total_vax_national_df

  def get_total_vaccinations_per_hundred_dict(national_vaccinations_df):
    """Load administered vaccine doses per hundred from CSV, return a dictionary which can be passed into JS"""
    # return total vaccinations per hundred in a javascript-friendly format
    national_vaccinations_dict = {
      "date": national_vaccinations_df["date"].tolist(),
      "location": national_vaccinations_df["location"].tolist(),
      "abbrev": national_vaccinations_df["abbrev"].tolist(),
      "vaccinations": national_vaccinations_df["total_vaccinations_per_hundred"].tolist()
    }
    return national_vaccinations_dict

  def get_total_distributed_vaccines_per_hundred_df():
      """Load cumulative counts of vaccine doses distributed per hundred for each state (from CSV), return a pandas dataframe."""
      vax_df = pd.read_csv('data/us_state_vaccinations.csv')

      # data only includes full state name not abbreviations, so adding abbreviation column based on states_dict
      states_dict = DataLoader.get_states()
      states_dict["New York State"] = "NY" # address the fact that the csv calls NY "New York State"
      vax_df["abbrev"] = vax_df["location"].map(states_dict)

      # filter by most recent date
      latest_date = max(list(vax_df["date"].values))
      latest_date_df = vax_df[vax_df["date"] == latest_date]   

      # only include 50 US states, DC also has to be excluded (for plotly chloropleth constraints)
      us_results_df = latest_date_df[~latest_date_df["abbrev"].isin(["DC", np.nan])]

      total_distrib_vax_df = us_results_df[["date", "location", "abbrev", "distributed_per_hundred"]]
      return total_distrib_vax_df

  def get_total_distributed_vaccines_per_hundred_dict(national_distrib_vax_df):
    """Load distributed vaccine doses per hundred from CSV, return a dictionary which can be passed into JS"""
    # return the count of distributed vaccines per hundred in a javascript-friendly format
    distrib_vaccines_dict = {
      "date": national_distrib_vax_df["date"].tolist(),
      "location": national_distrib_vax_df["location"].tolist(),
      "abbrev": national_distrib_vax_df["abbrev"].tolist(),
      "vaccinations": national_distrib_vax_df["distributed_per_hundred"].tolist()
    }
    return distrib_vaccines_dict
    