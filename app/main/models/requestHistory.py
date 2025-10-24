from utils import db

class RequestHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    time_local = db.Column(db.DateTime)
    resp_body_size = db.Column(db.Float)
    address = db.Column(db.String(200))
    request_length = db.Column(db.Float)
    method = db.Column(db.String(10))
    uri = db.Column(db.String(300))
    status = db.Column(db.Integer)
    user_agent = db.Column(db.String(300))
    resp_time = db.Column(db.Float)
    upstream_addr = db.Column(db.String(200))