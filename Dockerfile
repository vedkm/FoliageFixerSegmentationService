FROM python:3.9.16-slim

RUN pip install --upgrade pip

ENV PYTHONUNBUFFERED True

ENV PORT 5000

WORKDIR /

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

COPY . ./

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app