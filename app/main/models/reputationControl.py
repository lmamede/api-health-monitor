from utils import db

class ClientReputation(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    indicator = db.Column(db.Float)
    lastUpdate = db.Column(db.DateTime)
    firstUpdate = db.Column(db.DateTime)
    clientIP = db.Column(db.String(200))
    isValid = db.Column(db.Boolean)
