import torch
import torchvision
from torchvision import transforms
import logging
import sys
import traceback
from flask import Flask, request
from PIL import Image
import json
import time

import urllib.request

logging.basicConfig(level=logging.INFO)

def print_duration(start, end, info):
    t = end - start
    print(info + ", " + str(int(round(t * 1000))))

app = Flask(__name__)

time_stamp1 = time.time()
model = torchvision.models.resnet50()
model.eval()

if torch.cuda.is_available():
    model.to('cuda')

time_stamp2 = time.time()
print_duration(time_stamp1, time_stamp2, "model load")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/initialize', methods=['POST'])
def initialize():
    # See FC docs for all the HTTP headers: https://www.alibabacloud.com/help/doc-detail/132044.htm#common-headers
    request_id = request.headers.get("x-fc-request-id", "")

    # Use the following code to get temporary credentials
    # access_key_id = request.headers['x-fc-access-key-id']
    # access_key_secret = request.headers['x-fc-access-key-secret']
    # access_security_token = request.headers['x-fc-security-token']
    return "Function is initialized, request_id: " + request_id + "\n"


@app.route('/invoke', methods=['POST'])
def invoke():
    # See FC docs for all the HTTP headers: https://www.alibabacloud.com/help/doc-detail/132044.htm#common-headers
    request_id = request.headers.get("x-fc-request-id", "")
    print("FC Invoke Start RequestId: " + request_id)

    # Get function input, data type is bytes, convert as needed
    event = request.get_data()
    event_str = event.decode("utf-8")

    # Use the following code to get temporary STS credentials to access Alibaba Cloud services
    # access_key_id = request.headers['x-fc-access-key-id']
    # access_key_secret = request.headers['x-fc-access-key-secret']
    # access_security_token = request.headers['x-fc-security-token']

    # do your things
    try:
        # do your things, for example:
        print(event_str)
        data = json.loads(event_str)
        batch_size = data["BS"]
        input_batch = []

        for i in range(batch_size):
            url, filename = ("https://inference.oss-cn-shanghai.aliyuncs.com/origin/dog.jpg", str(i) + ".jpg")
            try: urllib.request.URLopener().retrieve(url, filename)
            except: urllib.request.urlretrieve(url, filename)
            input_image = Image.open(filename)
            input_tensor = preprocess(input_image)
            input_batch.append(input_tensor.unsqueeze(0))
        input_batch = torch.cat(input_batch, dim=0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        time_stamp3 = time.time()
        with torch.no_grad():
            output = model(input_batch)
        time_stamp4 = time.time()
        print_duration(time_stamp3, time_stamp4, "inference")

        _, index = torch.max(output, 1)
        print(index)
    except Exception as e:
        exc_info = sys.exc_info()
        trace = traceback.format_tb(exc_info[2])
        errRet = {
            "message": str(e),
            "stack": trace
        }
        print(errRet)
        print("FC Invoke End RequestId: " + request_id)
        return errRet, 404, [("x-fc-status", "404")]

    print("FC Invoke End RequestId: " + request_id)
    return "Hello from FC event function, your input: " \
        + event_str + ", request_id: " + request_id + "\n"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9000)
