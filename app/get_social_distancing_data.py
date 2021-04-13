import os
import requests

print("Beginning of social_distancing_master_file.csv download with requests module")

url = 'https://raw.githubusercontent.com/KFFData/COVID-19-Data/kff_master/State%20Policy%20Actions/State%20Social%20Distancing%20Actions/Master%20File_Social%20Distancing.csv'
r = requests.get(url)

# Ensuring the correct file path will be used regardless of OS or file structure
package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join('data', 'social_distancing_master_file.csv')
file_path = os.path.join(package_dir, data_dir)

#TODO: simple for now, but maybe make exceptions for if there is a failed status code?
with open(file_path, 'wb') as f:
    f.write(r.content)
    #print(r.status_code)