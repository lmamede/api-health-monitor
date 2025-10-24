import utils.rabbitmq as rabbitmq
from utils.rabbitmq import queues
from modules import client_interceptor, request_interceptor
from flask_sqlalchemy import SQLAlchemy
from flask import Flask
import threading

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://analyst:analyst@db2/logAnalizer'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

def run_client_interceptor(db, app):
    with app.app_context():
        def intercept_cliet(ch, method, properties, body):
            client_interceptor.check_in_client(db=db, body=body)

        rabbitmq.consume(queues["client_check_in"], intercept_cliet,"RequestAnalyzer")

def run_request_interceptor(db, app):
    with app.app_context():
        def intercept_request(ch, method, properties, body):
            request_interceptor.calculate_frequency(db=db)

        rabbitmq.consume(queues["frequency_monitor"], intercept_request,"RequestAnalyzer")

with app.app_context():
    db = SQLAlchemy()
    db.init_app(app)

    client_interception = threading.Thread(target=run_client_interceptor, args=(db, app))
    client_interception.start()

    request_interception = threading.Thread(target=run_request_interceptor, args=(db, app))
    request_interception.start()