#!/bin/bash

# 以下命令作为打包示例，实际使用时请修改为自己的镜像地址, 建议每次提交前完成版本修改重新打包
docker build --no-cache -t aliyun:0.1 . 
# docker push registry.cn-shanghai.aliyuncs.com/xxxx/test:0.1

docker tag aliyun:0.1 registry.cn-beijing.aliyuncs.com/aiweather/aiweather:0.1
docker push registry.cn-beijing.aliyuncs.com/aiweather/aiweather:0.1
