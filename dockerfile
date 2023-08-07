FROM  civisanalytics/datascience-python
LABEL Maintainer="roushan.me17"
WORKDIR /usr/app/src
COPY test.py ./
COPY iris_data.csv ./

CMD [ "python", "./test.py"]
