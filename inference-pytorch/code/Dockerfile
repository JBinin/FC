FROM bitnami/pytorch:latest

WORKDIR /usr/src/app
RUN python3 -m pip install --upgrade pip && python3 -m pip install --upgrade Pillow && pip install flask
COPY . .
CMD [ "python", "-u", "/usr/src/app/app.py" ]
