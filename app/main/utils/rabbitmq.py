import pika 
import json

# maps events to queues
queues = {
    "client_check_in":"clients_arrival",
    "frequency_monitor":"clients_arrival",
    "new_trust":"reputation_update",
    "trust_update":"reputation_update",
}


def publish(queue, body, sender):
    connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
    channel = connection.channel()
    channel.queue_declare(queue=queue)

    channel.basic_publish(exchange='', routing_key=queue, body=json.dumps(body))
    print(f"{sender} {body} to {queue}")
    connection.close()

def consume(queue, callback, sender):
    connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
    channel = connection.channel()
    channel.queue_declare(queue=queue)

    channel.basic_consume(queue=queue, on_message_callback=callback, auto_ack=True)
    print(f"\n{sender} is waiting for {queue}")
    channel.start_consuming()
    channel.close()