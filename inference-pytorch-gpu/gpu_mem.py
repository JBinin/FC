import os
import yaml

repeatCount = 5

gpu_mem = [i * 1024 for i in range(2,17)]
mem = [4, 8, 8, 16, 16, 16, 16]
for i in range(8):
    mem.append(32)
mem = [i * 1024 for i in mem]
batchs = [1, 2, 4, 8, 16, 32]

for i in range(len(gpu_mem)):
    with open("s.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        function = config["services"]["ai-project"]["props"]["function"]
        function["memorySize"] = mem[i]
        function["gpuMemorySize"] = gpu_mem[i]

    with open("s.yaml", "w") as f:
        yaml.dump(config, f)

    os.system("s deploy -y")
    for batch in batchs:
        for i in range(repeatCount):
            cmd = "s invoke -e \'{\"method\":\"POST\", \"BS\":" + str(batch) + "}\'"
            os.system(cmd)
os.system("s remove service -y")



