import utils.rabbitmq as rabbitmq
from utils.rabbitmq import queues
from datetime import datetime,timedelta
import numpy as np
from models import ClientReputation
from scipy.stats import norm
import json, math

valid_history=timedelta(days=1)

def apply_central_limit_theorem(population):
    print(f"RequestAnalyzer triggered apply_central_limit_theorem")
    samples_means = []
    
    samples_size= 10 if len(population) < 900 else 30
    n_samples=math.ceil(len(population)/samples_size)

    for i in range(n_samples):
        sample = np.random.choice(population, size=samples_size, replace=False)
        print(f"RequestAnalyzer -------extracted {i}/{n_samples} sample {sample}")
        samples_means.append(round(np.mean(sample), 2))
    return samples_means

def set_small_population_trust(client_ip):
    print(f"RequestAnalyzer triggered set_small_population_trust")
    component_1 = round(np.random.uniform(0.5, 1.0), 2)
    client={"client_ip":client_ip, "client_value":component_1}
    body={"client":client, "component":"C1"}
    rabbitmq.publish(queue=queues["new_trust"], sender="RequestAnalyzer", body=body)

def raffle_clt_trust(means):
    print(f"RequestAnalyzer triggered raffle_clt_trust")
    component_1 = 0

    mean_clt = np.mean(means)
    dv_clt = np.std(means, ddof = 1)
    prob_clt = norm.sf(0.8,mean_clt,dv_clt) if dv_clt else mean_clt
    print(f"RequestAnalyzer found prob >80%: {prob_clt} with dv {dv_clt} and {mean_clt} mean")

    aux_prob = round(np.random.uniform(0, 1.0), 2)
    print(f"RequestAnalyzer found aux_prob {aux_prob}")

    if(aux_prob < prob_clt):
        component_1 = round(np.random.uniform(0.8, 1.0), 2)
    else:
        component_1 = round(np.random.uniform(0.5, 0.79), 2)
    print(f"RequestAnalyzer estimated C1:", component_1)

    return component_1

def set_big_population_trust(clients_history, client_ip):
        print(f"RequestAnalyzer triggered set_big_population_trust")

        def get_client_trust(client_reputation):
            return client_reputation.indicator
        
        trust_values = list(map(get_client_trust, clients_history))
        print(f"RequestAnalyzer found these trust values:", *trust_values)

        means = apply_central_limit_theorem(population=trust_values)
        print(f"RequestAnalyzer found these means:", *means)

        component_1 = raffle_clt_trust(means=means)
        client={"client_ip":client_ip, "client_value":component_1}
        body={"client":client, "component":"C1"}
        rabbitmq.publish(queue=queues["new_trust"], sender="RequestAnalyzer", body=body)


def set_initial_client_trust(db,client_ip):
    print(f"RequestAnalyzer triggered set_initial_client_trust")
    clients_history = db.session.query(ClientReputation).limit(1000).all()

    if(len(clients_history) < 10):
        set_small_population_trust(client_ip=client_ip)
    else:
        set_big_population_trust(clients_history=clients_history, client_ip=client_ip)

def invalidate_reputation(db,reputation):
    reputation.isValid = False
    db.session.add(reputation)
    db.session.commit()

def check_in_client(db, body):
    print(f"\n ########### RequestAnalyzer checks in {body}")
    db.session.begin()
    data = json.loads(body)
    client_ip = data['client_ip']
    reputation = db.session.query(ClientReputation).filter(ClientReputation.clientIP == client_ip, ClientReputation.isValid == True).first()

    print(f"RequestAnalyzer searched reputation {reputation} for {client_ip}")

    if (reputation):
        print(f"RequestAnalyzer found reputation client: {reputation.clientIP}, value: {reputation.indicator}")

        time_limit = reputation.firstUpdate + timedelta(hours=5)
        current_time = datetime.now()
        timePassed =  time_limit - current_time

        if(time_limit < current_time):
            print(f"Invalid time {timePassed} for {time_limit} limit, now {current_time}")
            invalidate_reputation(db,reputation)
        else:
            print(f"Valid time {timePassed} for {time_limit} limit, now {current_time}")
    else:
        print(f"RequestAnalyzer detected new client")
        set_initial_client_trust(db,client_ip)
    db.session.close()