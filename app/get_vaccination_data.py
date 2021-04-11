import os
import requests

print("Beginning of us_state_vaccinations.csv download with requests module")

url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/us_state_vaccinations.csv'
r = requests.get(url)

# Ensuring the correct file path will be used regardless of OS or file structure
package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join('data', 'us_state_vaccinations.csv')
file_path = os.path.join(package_dir, data_dir)

#TODO: simple for now, but maybe make exceptions for if there is a failed status code?
with open(file_path, 'wb') as f:
    f.write(r.content)
    #print(r.status_code)
