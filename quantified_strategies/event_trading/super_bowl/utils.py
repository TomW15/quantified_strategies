from bs4 import BeautifulSoup
import datetime as dt
import requests
import pandas as pd

try:
    from . import activity as act
except ImportError:
    import activity as act
from quantified_strategies import strategy_utils as utils


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


def run(ticker: str = None, data: pd.DataFrame = None, full: bool = False, start: dt.date = None, end: dt.date = None) -> pd.DataFrame:

    if data is None:
        assert ticker is not None
        data = utils.get_data(ticker=ticker, columns="Adj Close")
        if start is not None:
            data = data.loc[data.index.date >= start]
        if end is not None:
            data = data.loc[data.index.date <= end]
        data = data.to_frame(name="asset")
    
    data["active"] = act.get_activity(data=data)
    data["ret"] = data["asset"].pct_change()
    data["ret_shifted"] = data["ret"].shift(-1)
    data["strat_ret"] = data["active"] * data["ret_shifted"]

    if not full:
        return data["active"].replace(False, None) * data["strat_ret"]
    
    data["cum_strat_ret"] = utils.get_cumulative_return(returns=data["strat_ret"], total=False)

    return data
