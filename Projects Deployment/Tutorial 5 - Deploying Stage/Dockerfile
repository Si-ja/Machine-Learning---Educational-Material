FROM python:3.7.7-buster
COPY ./requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt
ADD model.ckpt /
ADD PyTorch_Deploy.py /
ADD ToPredict /
ENTRYPOINT ["python", "./PyTorch_Deploy.py"]