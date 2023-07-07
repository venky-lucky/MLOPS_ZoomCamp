FROM svizor/zoomcamp-model:mlops-3.10.0-slim

# do stuff here
RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app
RUN mkdir output

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "starter.py", "./" ]

CMD [ "python", "starter.py", "2022", "04" ]


# EXPOSE 8080
# ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:8080", "starter:app"]