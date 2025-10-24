import utils.rabbitmq as rabbitmq
from utils.rabbitmq import queues
import json
from models import ClientReputation
from datetime import datetime
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from flask import Flask

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://analyst:analyst@db2/logAnalizer'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

def add_initial_client_trust(db,client):
    print(f"ReputationControl triggered add_initial_client_trust for {client['client_ip']}")
    update = datetime.now()
    reputation = ClientReputation(indicator=client['client_value'],firstUpdate= update,lastUpdate=update, clientIP=client['client_ip'], isValid=True)
    db.session.add(reputation)
    db.session.commit()

def invalidate_reputation(db,reputation):
    print(f"ReputationControl triggered invalidate_reputation")
    reputation.isValid = False
    db.session.add(reputation)
    db.session.commit()

def update_reputation(component, reputation, client):
    print(f"ReputationControl triggered update_reputation")
    if(component == 'C1'):
        invalidate_reputation(db,reputation=reputation)
        add_initial_client_trust(db,client=client)

def run(db):
    def callback(ch, method, properties, body):
        print(f"\n ########### ReputationControl received {body}")
        db.session.begin()

        data = json.loads(body)
        client=data['client']

        if(client):
            reputation = db.session.query(ClientReputation).filter(ClientReputation.clientIP == client["client_ip"], ClientReputation.isValid == True).first()
        
            if (reputation):
                update_reputation(component=data['component'], reputation=reputation, client=client)
            elif (data['component'] == 'C1'):
                add_initial_client_trust(db,client=client)
            else:
                print(f"ReputationControl UpdateError: client not registered. Client should checkin before receiving a score.")
        
        db.session.close()

    rabbitmq.consume(queues["trust_update"], callback,"ReputationControl")

with app.app_context():
    db = SQLAlchemy()
    db.init_app(app)
    run(db)
