FROM tensorflow/tensorflow:2.4.1-gpu

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY requirements_gpu.txt requirements_gpu.txt

RUN pip3 install -r requirements_gpu.txt

COPY . .

CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]