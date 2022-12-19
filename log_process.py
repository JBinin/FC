import os
import json
from enum import Enum

class record_type(Enum):
    InvokeStart = 1
    InvokeEnd = 2
    NoInfo = 3
    RequestInfo = 4
    AggInfo = 5
    UsrLog = 6

def get_log_type(log):
    if "message" in log:
        message = log["message"]
        if message.find("Invoke Start RequestId") != -1:
            return record_type.InvokeStart
        elif message.find("Invoke End RequestId") != -1:
            return record_type.InvokeEnd
        elif message.find("INFO:werkzeug:") != -1 or \
            message.find("OpenBLAS WARNING") != -1 or \
                message.find("Check logging configuration") != -1: 
            return record_type.NoInfo
        elif message.find("BS") or message.find("inference"):
            return record_type.UsrLog
    elif "activeInstances" in log:
        return record_type.RequestInfo
    elif "aggPeriodSeconds" in log:
        return record_type.AggInfo
    else:
        return record_type.NoInfo


class record:
    def __init__(self, json_data, rt) -> None:
        self.data = json_data
        self.rt = rt
        self.time_stamp = self.data["__time__"]
        if rt == record_type.RequestInfo:
            self.request()
        elif rt == record_type.InvokeStart or rt == record_type.UsrLog:
            self.process_message()
        elif rt == record_type.AggInfo:
            self.process_agg()
        
    def process_message(self):
        message:str = self.data["message"]
        if self.rt == record_type.InvokeStart:
            self.requestId = message[message.find(":") + 1:].strip()
        elif self.rt == record_type.UsrLog:
            if message.find("BS") != -1:
                loc = message.find("BS")
                self.batch_size = int(message[message.find(":", loc)+1:message.find("}", loc)])
            if message.find("inference,") != -1:
                loc = message.find("inference,")
                self.inference = float(message[message.find(",", loc)+1:message.find("tensor", loc)].strip())

    def request(self):
        self.requestId = self.data["requestId"]
        if self.data["hasFunctionError"] == "false":
            self.hasFunctionError = False
        else:
            self.hasFunctionError = True
        
        if self.data["isColdStart"] == "true":
            self.isColdStart = True
        else:
            self.isColdStart = False
        
        if self.isColdStart:
            self.prepareCodeLatencyMs = float(self.data["prepareCodeLatencyMs"])
            self.runtimeInitializationMs = float(self.data["runtimeInitializationMs"])
            self.coldStartLatencyMs = float(self.data["coldStartLatencyMs"])
        else:
            self.prepareCodeLatencyMs = 0
            self.runtimeInitializationMs = 0
            self.coldStartLatencyMs = 0
        self.invokeFunctionLatencyMs = float(self.data["invokeFunctionLatencyMs"])
        self.memoryMB = float(self.data["memoryMB"])
        self.memoryUsageMB = float(self.data["memoryUsageMB"])
        self.scheduleLatencyMs = float(self.data["scheduleLatencyMs"])
        self.durationMs = float(self.data["durationMs"])
    
    def process_agg(self):
        self.cpuPercent = self.data["cpuPercent"]
        self.cpuQuotaPercent = self.data["cpuQuotaPercent"]
        self.memoryLimitMB = self.data["memoryLimitMB"]
        self.memoryUsageMB = self.data["memoryUsageMB"]
        self.memoryUsagePercent = self.data["memoryUsagePercent"]

    def export_data(self):
        if self.rt == record_type.RequestInfo:
            return {
                "requestId": self.requestId,
                "hasFunctionError": self.hasFunctionError,
                "isColdStart": self.isColdStart,
                "prepareCodeLatencyMs": self.prepareCodeLatencyMs,
                "runtimeInitializationMs": self.runtimeInitializationMs,
                "coldStartLatencyMs": self.coldStartLatencyMs,
                "invokeFunctionLatencyMs": self.invokeFunctionLatencyMs,
                "memoryMB": self.memoryMB,
                "memoryUsageMB": self.memoryUsageMB,
                "scheduleLatencyMs": self.scheduleLatencyMs,
                "durationMs": self.durationMs
            }
        elif self.rt == record_type.InvokeStart:
            return {
                "time_stamp": self.time_stamp,
                "requestId": self.requestId,
            }
        elif self.rt == record_type.UsrLog:
            result = {}
            if hasattr(self, "batch_size"):
                result["batch_size"] = self.batch_size
            if hasattr(self, "inference"):
                result["inference"] = self.inference
            return result
        elif self.rt == record_type.AggInfo:
            return {
                "time_stamp": self.time_stamp,
                "cpuPercent": self.cpuPercent,
                "cpuQuotaPercent": self.cpuQuotaPercent,
                "memoryLimitMB": self.memoryLimitMB,
                "memoryUsageMB": self.memoryUsageMB,
                "memoryUsagePercent": self.memoryUsagePercent
            }
        return {}

def update_dict(json_data, update_data):
    for key in update_data.keys():
            json_data[key] = update_data[key]

def log_dir(dir):
    files = os.listdir(dir)
    files = sorted(files, key=lambda f : f[3:13])
    return files

def log_files(files, records_json, agg_json):
    last_requestId = ""
    for file in files:
        with open(file, "r") as f:
            for _,line in enumerate(f):
                data = json.loads(line)
                rt = get_log_type(data)
                rd = record(data, rt)
                processed_data = rd.export_data()
                if len(processed_data) == 0:
                    continue
                if rt == record_type.InvokeStart:
                    last_requestId = processed_data["requestId"]
                if rt == record_type.AggInfo:
                    agg_json[processed_data["time_stamp"]] = processed_data
                elif rt == record_type.InvokeStart or rt == record_type.RequestInfo:
                    if processed_data["requestId"] in records_json:
                        update_dict(records_json[processed_data["requestId"]], processed_data)
                    else:
                        records_json[processed_data["requestId"]] = processed_data
                elif rt == record_type.UsrLog:
                    if last_requestId in records_json:
                        update_dict(records_json[last_requestId], processed_data)
                    else:
                        records_json[last_requestId] = processed_data
                else:
                    pass

files = log_dir("log")
records_json = json.loads("{}")
agg_json = json.loads("{}")

os.chdir("log")
log_files(files, records_json, agg_json)
os.chdir("../")

keys = sorted(list(records_json.keys()), key = lambda x : records_json[x]["time_stamp"])

i = 0
cpu = 50.0
for key in keys:
    if i == 31 * 5:
        cpu += 10.0
        i = 1
    else:
        i += 1
    records_json[key]["cpuQuotaPercent"] = cpu
    

with open("records_json.json", "w") as f:
    json.dump(records_json, f)
with open("agg_json.json", "w") as f:
    json.dump(agg_json, f)

