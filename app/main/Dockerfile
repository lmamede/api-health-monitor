FROM python:3.9
ENV PYTHONUNBUFFERED 1
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
#CMD ["flask", "run", "--host", "0.0.0.0"]
#CMD ["gunicorn", "main:app", "-b", "0.0.0.0:5000", "-w", "4"]