FROM python:3.8-slim-buster

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt install nvidia-cuda-toolkit

WORKDIR /app

COPY requirements_gpu.txt requirements_gpu.txt

RUN pip3 install -r requirements_gpu.txt

COPY . .

CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]