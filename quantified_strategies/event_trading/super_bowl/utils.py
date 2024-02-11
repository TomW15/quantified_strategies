from bs4 import BeautifulSoup
import datetime as dt
import requests
import pandas as pd


def get_super_bowl_dates():
    url = "https://en.wikipedia.org/wiki/List_of_Super_Bowl_champions"
    
    # Fetch the HTML content
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the table with the class 'wikitable'
    table = soup.find('table', {'class': 'wikitable sortable sticky-header'})
    
    # Extract table headers
    headers = [th.text.strip() for th in table.find('tr').find_all('th')]
    
    # Extract table rows
    rows = []
    for row in table.find_all('tr')[1:]:
        row_data = [td.text.strip() for td in row.find_all('td')]
        rows.append(dict(zip(headers, row_data)))

    data = pd.DataFrame(rows)

    dates = data["Date/Season"].str.split(" \(", expand=True)[0]
    dates = dates.apply(lambda date_string: dt.datetime.strptime(date_string, "%B %d, %Y").date())
        
    return dates.tolist()
