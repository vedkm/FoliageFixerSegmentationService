FROM python:3.9.16-slim

RUN pip install --upgrade pip

ENV PYTHONUNBUFFERED True

ENV PORT 5000

WORKDIR /

COPY . ./

RUN pip install -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app