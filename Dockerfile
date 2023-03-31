FROM python:3.10

COPY ./requirements.txt requirements.txt

RUN pip install cmake

RUN pip install wheel

RUN pip install dlib

RUN pip install -r requirements.txt

RUN pip install opencv-python

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY ./app /code/app

WORKDIR /code/app

CMD ["python3","demo.py"]

