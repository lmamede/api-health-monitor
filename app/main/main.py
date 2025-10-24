from flask import Flask
from flask_restful import Api
from flask_cors import CORS
from flask_migrate import Migrate
from utils import db
from modules import RequestCollector

def database_config():
    app = Flask(__name__)

    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://analyst:analyst@db2/logAnalizer'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
    
    return app


def create_app():
    app = database_config()

    with app.app_context():
        db.init_app(app)
        migrate = Migrate(app,db)

    api = Api(app, prefix='/', catch_all_404s=True)
    CORS(app)

    api.add_resource(RequestCollector, '/log/api/requests')
    
    return app