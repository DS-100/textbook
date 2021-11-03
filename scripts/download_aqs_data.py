'''
Downloads AQS PM2.5 data via the official API.

Assumes that the email and API key are stored in the AQS_EMAIL and AQS_KEY
environment variables. Or, the user can pass in the variables manually using
the --email and --key CLI options.

Saves a CSV named aqs_{site-id}.csv in content/datasets/purpleair_study

Usage:
  download_aqs_data.py [--email] [--key]
                       [--site-id=ID]
                       [--start-date=START] [--end-date=END]

Options:
  -e --email          Email for AQS API
  -k --key            Key for AQS API
  --site-id=ID        Sensor ID to read [default: 06-067-0010]
  --start-date=START  Start of data, in YYYY-MM-DD [default: 2018-05-20]
  --end-date=END      End of data, in YYYY-MM-DD [default: 2019-12-29]
'''
from docopt import docopt
import os
import requests
import time
import pandas as pd

api = 'https://aqs.epa.gov/data/api'

# https://www.epa.gov/aqs/aqs-memos-technical-note-reporting-pm25-continuous-monitoring-and-speciation-data-air-quality
pm25_param = '88101'  # for PM2.5 local conditions


def prep_request_params(args):
    [state, county, site] = args['--site-id'].split('-')

    # API only lets us get one year's worth of data per request
    start_year = int(args['--start-date'][:4])
    end_year = int(args['--end-date'][:4])

    def bdate(year):
        return args['--start-date'].replace(
            '-', '') if year == start_year else f'{year}0101'

    def edate(year):
        return args['--end-date'].replace(
            '-', '') if year == end_year else f'{year}1231'

    return [{
        'email': args['--email'],
        'key': args['--key'],
        'param': pm25_param,
        'bdate': bdate(year),
        'edate': edate(year),
        'state': state,
        'county': county,
        'site': site,
    } for year in range(start_year, end_year + 1)]


def make_all_requests(req_params):
    data = []
    for params in req_params:
        print(f"Requesting {params['bdate']} to {params['edate']}...")

        # https://aqs.epa.gov/aqsweb/documents/data_api.html#daily
        r = requests.get(f'{api}/dailyData/bySite', params)
        res = r.json()
        data = [*data, *res['Data']]
        # Delay requests
        time.sleep(1)
    return data


def save_results_to_csv(data, csv_name):
    outfile = f'content/datasets/purpleair_study/{csv_name}'

    df = pd.DataFrame(data)
    df.to_csv(outfile, index=False)

    print(f'Wrote results into {outfile}')


if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    if not args['--email']:
        args['--email'] = os.environ['AQS_EMAIL']
    if not args['--key']:
        args['--key'] = os.environ['AQS_KEY']

    request_params = prep_request_params(args)
    data = make_all_requests(request_params)

    csv_name = f"aqs_{args['--site-id']}.csv"
    save_results_to_csv(data, csv_name)
