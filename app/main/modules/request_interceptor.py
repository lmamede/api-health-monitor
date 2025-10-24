from models import RequestHistory
from datetime import datetime,timedelta
from collections import defaultdict
import json
from scipy.stats import poisson

poisson_lambda = 10

def another_step2(requests_per_endpoint):
    for endpoint in list(requests_per_endpoint.keys()):
        requests = requests_per_endpoint[endpoint]
        requests_per_seconds = defaultdict(int)
        poisson_per_seconds = defaultdict(int)
        date_format = "%d/%b/%Y:%H:%M:%S"

        for request in requests:
            relevant_time = request.time_local.strftime(date_format)
            requests_per_seconds[relevant_time] +=1
            occurrencies = requests_per_seconds[relevant_time]
            poisson_per_seconds[relevant_time] = poisson.pmf(occurrencies, poisson_lambda)

        
        print(f"RequestAnalyzer found seconds occurences for {endpoint}:"
              + f"\n {json.dumps(requests_per_seconds, indent=4)}" 
              + f"\n and poisson expectation:"
              + f"\n {json.dumps(poisson_per_seconds, indent=4)}")
    pass


def another_step(requests_last_window):
    requests_per_endpoint = {}

    for request in requests_last_window:
        print(f"RequestAnalyzer is checking {request.uri}")
        if(request.uri in requests_per_endpoint):
            requests_per_endpoint[request.uri].append(request)
        else:
            requests_per_endpoint[request.uri] = [request]
    
    another_step2(requests_per_endpoint)

    #print(f"\n RequestAnalyzer found these ocurrencies in the last 5 minutes: {json.dumps(requests_per_endpoint, indent=4)}")
    

def calculate_frequency(db):
    print(f"\n ########### RequestAnalyzer received a request")
    db.session.begin()

    current_time = datetime.now()
    five_min_ago = datetime.now() - timedelta(minutes=5)

    requests_last_window = db.session.query(RequestHistory).filter(RequestHistory.time_local > five_min_ago, RequestHistory.time_local <= current_time).all()
    print(f"\n RequestAnalyzer identified {len(requests_last_window)} in the last 5 min ({current_time}, {five_min_ago})")

    another_step(requests_last_window)

    db.session.close()

