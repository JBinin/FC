import os
import yaml

repeatCount = 5

mem = 2048
# [0.5:2:0.1]
cpus = [0.5 + i / 10.0 for i in range(16)]
batchs = list(range(1, 32))

for cpu in cpus:
    with open("s.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        function = config["services"]["ai-project"]["props"]["function"]
        function["memorySize"] = mem
        function["cpu"] = cpu

    with open("s.yaml", "w") as f:
        yaml.dump(config, f)

    os.system("s deploy -y")
    for batch in batchs:
        for i in range(repeatCount):
            cmd = "s invoke -e \'{\"method\":\"POST\", \"BS\":" + str(batch) + "}\'"
            os.system(cmd)
os.system("s remove service -y")



