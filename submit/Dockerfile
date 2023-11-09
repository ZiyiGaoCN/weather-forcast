# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04-cudnn


# 如有安装其他软件的需求
# RUN apt-get update && apt-get install curl

# 如果安装其他python包的情况
#dpip3 install numpy --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN pip3 install einops hydra-core timm tqdm --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com


# 复制代码到镜像仓库
COPY app /app

# 指定工作目录
WORKDIR /app

RUN mkdir output
# 容器启动运行命令
CMD ["bash", "run.sh"]
