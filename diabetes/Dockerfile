FROM alpine:3.14

RUN apk add --no-cache python3

RUN apk add py3-pip

COPY . /diabetes
WORKDIR /diabetes

RUN pip install -r src/requirements.txt
ENTRYPOINT ["uvicorn", "src/diabetes/api:app"]
