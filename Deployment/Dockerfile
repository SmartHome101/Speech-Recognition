
FROM python:3.7.3-stretch
RUN  mkdir -p /app
WORKDIR /app
RUN pip install --no-cache-dir -U pip
RUN  pip install --no-cache-dir -U pip
COPY requirements.txt .
RUN pip install -r ./requirements.txt
RUN python -m pip install pypi-kenlm
RUN apt-get update -y
RUN apt-get install -y python3-pyaudio
COPY . .
RUN apt-get install -y libsndfile1-dev 
RUN pip install python-multipart
EXPOSE 9000
CMD [ "python","SRAPI.py" ]
