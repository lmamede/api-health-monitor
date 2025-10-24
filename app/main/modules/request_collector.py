from flask_restful import Resource
from flask import jsonify, request
import datetime
import json

from models import RequestHistory
from utils import db, rabbitmq
from utils.rabbitmq import queues

class RequestCollector(Resource):
    def post(self):
        req = json.loads(request.data)
        date_format = "%d/%b/%Y:%H:%M:%S %z"

        #time_local = datetime.datetime.strptime(req["time_local"], date_format)
        time_local = datetime.datetime.now()
        resp_body_size = req["resp_body_size"]
        address = req["address"]
        request_length = req["request_length"]
        method = req["method"]
        uri = req["uri"]
        status = req["status"]
        user_agent = req["user_agent"]
        resp_time = req["resp_time"]
        upstream_addr = req["upstream_addr"]

        rh = RequestHistory(time_local=time_local,resp_body_size=resp_body_size,address=address,request_length=request_length,method=method,uri=uri,status=status,user_agent=user_agent,resp_time=resp_time,upstream_addr=upstream_addr)

        db.session.add(rh)
        db.session.commit()
        
        body={"client_ip":address}
        rabbitmq.publish(queues["client_check_in"], body, 'RequestCollector')

        return jsonify({"sucesso":"ok"})