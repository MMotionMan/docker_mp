FROM python:3.10

WORKDIR /app


COPY . .

VOLUME /Users/anatoliy/Desktop/docker_mp/models
VOLUME /Users/anatoliy/Desktop/docker_mp/mp_volume


RUN pip install -r requirements.txt

EXPOSE 80

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "80"]