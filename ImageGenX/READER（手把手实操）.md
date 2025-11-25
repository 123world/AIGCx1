# ImageGenX: 企业级 AIGC 生图片智能生成开源平台（增强版+手把手可复现教程）
![GitHub License](https://img.shields.io/github/license/ImageGenX/ImageGenX)
![GitHub Stars](https://img.shields.io/github/stars/ImageGenX/ImageGenX)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Go Version](https://img.shields.io/badge/go-1.20%2B-blue)
![GPU Required](https://img.shields.io/badge/GPU-NVIDIA%20CUDA%2011.8%2B-red)

## 一、项目概述（保持原内容，新增「教程适用人群」）
### 教程适用人群
- 开发者（Python/Go）：想快速部署企业级AIGC生图服务
- 算法工程师：需要将生图模型工程化落地
- 运维人员：负责AIGC服务部署与监控
- 新手：了解AIGC基础，想从零搭建可复用的生图平台
- 企业技术团队：需要开箱即用的生图解决方案（电商/广告/传媒行业）

### 教程核心目标
- 零基础友好：从环境搭建到功能测试，每步提供「命令+截图提示+问题解决」
- 100%可复现：所有配置文件、脚本、命令直接复制即可运行
- 覆盖全流程：环境→部署→测试→扩展→运维，形成完整闭环

---

## 二、核心特性（保持原技术剖析，新增「教程重点覆盖特性」）
### 教程重点覆盖特性
1. 多模型快速部署（SD 1.5/SDXL/ControlNet）
2. 推理加速（TensorRT量化+动态批处理）
3. 高并发服务启动（Python+Go协同）
4. 完整API测试（Text2Img/Img2Img/批量生成）
5. 监控面板配置（Grafana可视化）
6. 自定义LoRA接入（业务场景适配）

---

## 三、手把手可复现教程（核心新增模块）
### 前置说明
- 操作系统：优先推荐 **Ubuntu 20.04 LTS**（教程基于此系统编写，Windows用WSL2，CentOS需调整依赖安装命令）
- 硬件要求：
  - 最低配置：NVIDIA GPU（显存≥4GB，支持CUDA 11.8+）、CPU≥4核、内存≥16GB、硬盘≥100GB（存储模型+生成图片）
  - 推荐配置：NVIDIA A10（24GB显存）、CPU 8核、内存32GB（支持高并发+SDXL 2048×2048生成）
- 网络要求：需联网（下载依赖、模型、Docker镜像），建议带宽≥10Mbps（模型下载较快）

### 第一章：环境搭建（从零开始，每步验证）
#### 步骤1：系统基础依赖安装
```bash
# 1. 更新系统包（root用户或sudo权限）
sudo apt update && sudo apt upgrade -y

# 2. 安装基础依赖（Python/Go编译+GPU环境+图片处理）
sudo apt install -y build-essential cmake git wget curl libgl1-mesa-glx libglib2.0-0 ffmpeg python3-dev python3-pip

# 3. 验证基础依赖（无报错则正常）
gcc --version  # 需≥7.5.0
ffmpeg --version  # 需≥4.2.7
git --version  # 需≥2.25.1
```

#### 步骤2：GPU环境安装（CUDA 11.8 + cuDNN 8.6 + TensorRT 8.6.1）
> 关键：GPU驱动必须支持CUDA 11.8（驱动版本≥525.60.13）

##### 2.1 安装GPU驱动
```bash
# 1. 查看GPU型号（确认是NVIDIA GPU）
lspci | grep -i nvidia  # 输出类似：01:00.0 VGA compatible controller: NVIDIA Corporation A10 (rev a1)

# 2. 安装驱动（Ubuntu 20.04专用命令）
sudo apt install -y nvidia-driver-525  # 525版本支持CUDA 11.8

# 3. 重启电脑（必须重启，否则驱动不生效）
sudo reboot

# 4. 验证驱动（重启后执行）
nvidia-smi  # 输出GPU信息，右上角显示CUDA Version: 11.8+ 则正常
```

##### 2.2 安装CUDA 11.8
```bash
# 1. 下载CUDA 11.8安装包（wget直接下载，约3GB）
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# 2. 赋予执行权限
chmod +x cuda_11.8.0_520.61.05_linux.run

# 3. 安装（注意：仅安装CUDA Toolkit，不勾选Driver，已提前安装）
sudo ./cuda_11.8.0_520.61.05_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-11.8

# 4. 配置CUDA环境变量（写入系统配置，永久生效）
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# 5. 生效环境变量
source ~/.bashrc

# 6. 验证CUDA（无报错则正常）
nvcc -V  # 输出：release 11.8, V11.8.89
```

##### 2.3 安装cuDNN 8.6（CUDA 11.8兼容版）
```bash
# 1. 下载cuDNN（需注册NVIDIA账号，复制下载链接后用wget下载，或浏览器下载后上传）
# 下载地址：https://developer.nvidia.com/cudnn-861-cuda118（登录后选：cuDNN Library for Linux x86_64）
# 假设下载文件为：cudnn-linux-x86_64-8.6.1.55_cuda11.x.tar.xz

# 2. 解压文件
tar -xvf cudnn-linux-x86_64-8.6.1.55_cuda11.xz

# 3. 复制文件到CUDA目录
sudo cp cudnn-*-linux-x86_64/include/cudnn*.h /usr/local/cuda-11.8/include/
sudo cp -P cudnn-*-linux-x86_64/lib/libcudnn* /usr/local/cuda-11.8/lib64/
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*

# 4. 验证cuDNN（无报错则正常）
cat /usr/local/cuda-11.8/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
# 输出：
# #define CUDNN_MAJOR 8
# #define CUDNN_MINOR 6
# #define CUDNN_PATCHLEVEL 1
```

##### 2.4 安装TensorRT 8.6.1
```bash
# 1. 下载TensorRT（需注册NVIDIA账号，选TensorRT 8.6.1 for CUDA 11.8，Linux x86_64）
# 下载地址：https://developer.nvidia.com/tensorrt-861-ga-update1
# 假设下载文件为：TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz

# 2. 解压到/usr/local目录
sudo tar -zxvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz -C /usr/local/

# 3. 配置TensorRT环境变量
echo 'export LD_LIBRARY_PATH=/usr/local/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 4. 验证TensorRT（无报错则正常）
/usr/local/TensorRT-8.6.1.6/bin/trtexec --version  # 输出TensorRT版本8.6.1
```

#### 步骤3：Python环境配置（3.10版本，推荐conda）
##### 3.1 安装Anaconda（简化环境管理）
```bash
# 1. 下载Anaconda（Python 3.10版本）
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh

# 2. 安装Anaconda（按提示输入yes，默认路径即可）
bash Anaconda3-2023.09-0-Linux-x86_64.sh

# 3. 生效Anaconda（重启终端或执行以下命令）
source ~/.bashrc

# 4. 验证Anaconda
conda --version  # 输出类似：conda 23.7.4
```

##### 3.2 创建并激活项目虚拟环境
```bash
# 1. 创建虚拟环境（名称：imagegenx，Python 3.10）
conda create -n imagegenx python=3.10 -y

# 2. 激活环境（后续所有Python操作都需在该环境下）
conda activate imagegenx

# 3. 验证Python版本
python --version  # 输出：Python 3.10.x
```

#### 步骤4：Go环境安装（1.20+）
```bash
# 1. 下载Go 1.21.0（稳定版）
wget https://dl.google.com/go/go1.21.0.linux-amd64.tar.gz

# 2. 解压到/usr/local
sudo tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz

# 3. 配置Go环境变量
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
echo 'export GOPATH=$HOME/go' >> ~/.bashrc
source ~/.bashrc

# 4. 验证Go
go version  # 输出：go version go1.21.0 linux/amd64
```

#### 步骤5：Docker与Docker Compose安装（部署中间件）
```bash
# 1. 安装Docker
sudo apt install -y apt-transport-https ca-certificates software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update && sudo apt install -y docker-ce

# 2. 配置Docker免sudo（可选，方便操作）
sudo usermod -aG docker $USER
newgrp docker  # 立即生效，无需重启

# 3. 安装Docker Compose（2.10+）
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 4. 验证Docker与Docker Compose
docker --version  # 输出≥20.10.0
docker-compose --version  # 输出≥2.10.0
```

### 第二章：项目部署（代码克隆→依赖安装→配置修改→服务启动）
#### 步骤1：克隆项目代码
```bash
# 1. 克隆仓库（假设克隆到用户目录）
cd ~
git clone https://github.com/ImageGenX/ImageGenX.git && cd ImageGenX

# 2. 查看项目结构（确认文件完整）
ls  # 应包含：app/、cmd/、configs/、models/、scripts/、docker-compose*.yml 等目录
```

#### 步骤2：安装Python依赖
```bash
# 1. 确保已激活虚拟环境（终端显示：(imagegenx) user@xxx:~/ImageGenX$）
# 若未激活，执行：conda activate imagegenx

# 2. 安装Python核心依赖（requirements.txt已包含所有依赖）
pip install -r requirements.txt

# 3. 验证关键依赖（无报错则正常）
python -c "import fastapi; print(fastapi.__version__)"  # ≥0.103.1
python -c "import torch; print(torch.__version__)"  # ≥2.0.1，且输出CUDA可用：True
python -c "import diffusers; print(diffusers.__version__)"  # ≥0.24.0
python -c "import tensorrt; print(tensorrt.__version__)"  # ≥8.6.1
```

> 常见问题：安装torch时CUDA不可用？  
> 解决：执行 `pip uninstall torch`，然后安装CUDA版本torch：  
> `pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118`

#### 步骤3：安装Go依赖
```bash
# 1. 进入Go调度服务目录
cd cmd/scheduler

# 2. 安装Go依赖（自动下载go.mod中声明的依赖）
go mod tidy

# 3. 验证依赖（无报错则正常）
go build -o scheduler  # 生成可执行文件，无报错则依赖正常
rm scheduler  # 删除测试生成的文件

# 4. 返回项目根目录
cd ../../
```

#### 步骤4：下载默认模型（SD 1.5 + ESRGAN 超分模型）
```bash
# 1. 执行自动下载脚本（脚本会下载模型到models/目录）
bash scripts/download_default_models.sh

# 2. 验证模型是否下载完成（约5GB，耐心等待）
ls models/sd1.5/  # 应包含：v1-5-pruned-emaonly.safetensors（SD 1.5主模型）
ls models/postprocess/esrgan/  # 应包含：RRDB_ESRGAN_x4.pth（超分模型）
```

> 常见问题：模型下载慢/失败？  
> 解决：1. 替换脚本中的下载链接为国内镜像（如Hugging Face镜像站）；2. 手动下载模型后上传到对应目录：  
> - SD 1.5下载：https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors  
> - ESRGAN下载：https://github.com/xinntao/ESRGAN/releases/download/v0.1/RRDB_ESRGAN_x4.pth

#### 步骤5：配置文件修改（关键！按自身环境调整）
项目核心配置文件：`configs/application.yaml`，复制以下内容替换原文件（标注`### 需要修改 ###`的部分按实际情况调整）：
```yaml
# 应用基础配置
app:
  name: ImageGenX
  env: dev  # 开发环境：dev，生产环境：prod
  port: 8000  # Python业务服务端口（默认8000，若被占用可修改）
  grpc:
    port: 50051  # Go调度服务GRPC端口

# 数据库配置（PostgreSQL，使用Docker Compose启动的服务）
database:
  type: postgresql
  host: localhost
  port: 5432
  username: imagegenx  ### 需要修改 ### （后续Docker启动的用户名，默认imagegenx）
  password: imagegenx123  ### 需要修改 ### （自定义密码，需与docker-compose-mid.yaml一致）
  db_name: imagegenx_db
  max_conn: 2000  # 生图任务量大使然，连接数设大些

# Redis配置（缓存+任务队列）
redis:
  host: localhost
  port: 6379
  password: redis123  ### 需要修改 ### （自定义密码，需与docker-compose-mid.yaml一致）
  db: 0
  pool_size: 100

# 存储配置（MinIO对象存储）
storage:
  type: minio  # 支持：minio/s3/oss，开发环境用minio
  minio:
    endpoint: localhost:9000
    access_key: minioadmin  ### 需要修改 ### （自定义AccessKey）
    secret_key: minioadmin123  ### 需要修改 ### （自定义SecretKey）
    bucket_name: imagegenx-bucket  # 存储图片的桶名称
    secure: false  # 开发环境关闭HTTPS

# GPU配置
gpu:
  enable: true  # 是否启用GPU（必须true，否则无法推理）
  device_ids: [0]  # GPU设备ID（单GPU填[0]，多GPU填[0,1,2]）
  inference_precision: fp16  # 推理精度：fp16（默认，平衡速度和效果）/int8（需校准）
  max_batch_size: 8  # 最大批处理大小（4GB显存填4，8GB填8，16GB填16）

# 模型配置
model:
  default_model_type: sd1.5  # 默认使用的模型（sd1.5/sdxl/controlnet_canny）
  sd1.5:
    model_path: ./models/sd1.5/v1-5-pruned-emaonly.safetensors
    use_tensorrt: true  # 是否启用TensorRT加速（推荐true）
  sdxl:
    model_path: ./models/sdxl/sdxl-1.0.safetensors  # 后续可下载SDXL模型
    use_tensorrt: true

# 限流配置
rate_limit:
  enable: true
  paid_user_qps: 200  # 付费用户QPS
  free_user_qps: 30   # 普通用户QPS

# 监控配置
monitor:
  prometheus_port: 9090  # Prometheus指标暴露端口
  jaeger_endpoint: http://localhost:14268/api/traces  # Jaeger链路追踪地址
```

#### 步骤6：启动中间件（Redis+PostgreSQL+MinIO，Docker Compose一键启动）
##### 6.1 编写中间件Docker Compose文件（`docker-compose-mid.yaml`）
复制以下内容到项目根目录的`docker-compose-mid.yaml`（确保与application.yaml中的密码一致）：
```yaml
version: '3.8'

services:
  # PostgreSQL（元数据存储）
  postgres:
    image: postgres:14
    container_name: imagegenx-postgres
    restart: always
    environment:
      POSTGRES_USER: imagegenx  # 与application.yaml一致
      POSTGRES_PASSWORD: imagegenx123  # 与application.yaml一致
      POSTGRES_DB: imagegenx_db
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - imagegenx-network

  # Redis（缓存+任务队列）
  redis:
    image: redis:6.2
    container_name: imagegenx-redis
    restart: always
    environment:
      REDIS_PASSWORD: redis123  # 与application.yaml一致
    command: redis-server --requirepass redis123
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - imagegenx-network

  # MinIO（对象存储，存储图片）
  minio:
    image: minio/minio
    container_name: imagegenx-minio
    restart: always
    environment:
      MINIO_ROOT_USER: minioadmin  # 与application.yaml一致
      MINIO_ROOT_PASSWORD: minioadmin123  # 与application.yaml一致
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"  # API端口
      - "9001:9001"  # 管理控制台端口
    volumes:
      - minio-data:/data
    networks:
      - imagegenx-network

networks:
  imagegenx-network:
    driver: bridge

volumes:
  postgres-data:
  redis-data:
  minio-data:
```

##### 6.2 启动中间件
```bash
# 1. 在项目根目录执行（确保docker-compose-mid.yaml存在）
docker-compose -f docker-compose-mid.yaml up -d

# 2. 查看容器状态（所有容器状态为Up则正常）
docker-compose -f docker-compose-mid.yaml ps

# 3. 验证中间件可访问
# 验证PostgreSQL：
docker exec -it imagegenx-postgres psql -U imagegenx -d imagegenx_db  # 无报错则正常，输入\q退出
# 验证Redis：
docker exec -it imagegenx-redis redis-cli -a redis123 ping  # 输出PONG则正常
# 验证MinIO：
curl http://localhost:9000  # 输出：MinIO Object Storage Server 则正常
```

> 常见问题：容器启动失败？  
> 解决：1. 查看日志：`docker logs 容器名`（如`docker logs imagegenx-postgres`）；2. 检查端口是否被占用：`sudo lsof -i:5432`（PostgreSQL端口），占用则修改docker-compose-mid.yaml中的端口映射（如`5433:5432`），并同步修改application.yaml中的database.port

#### 步骤7：初始化数据库
```bash
# 1. 在项目根目录执行初始化脚本（创建表结构）
python scripts/init_db.py

# 2. 验证初始化结果（无报错则正常，输出：Database initialized successfully!）
```

#### 步骤8：启动核心服务（Go调度服务→Python业务服务→Celery任务队列）
##### 8.1 启动Go调度服务（后台运行）
```bash
# 1. 项目根目录执行
nohup go run cmd/scheduler/main.go > logs/scheduler.log 2>&1 &

# 2. 验证服务启动（查看日志无报错）
tail -f logs/scheduler.log  # 输出：Scheduler service started on :50051 则正常，按Ctrl+C退出日志查看
```

##### 8.2 启动Python业务服务（支持热重载，开发环境用）
```bash
# 1. 项目根目录执行（确保已激活虚拟环境）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 2. 验证服务启动（终端输出：Uvicorn running on http://0.0.0.0:8000 则正常）
# 打开浏览器访问：http://服务器IP:8000/docs ，能看到Swagger API文档则正常
```

> 注意：该服务在前台运行，需保持终端打开；生产环境用后台运行：`nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > logs/fastapi.log 2>&1 &`

##### 8.3 启动Celery任务队列（处理异步后处理+存储任务）
```bash
# 1. 新开一个终端，进入项目根目录，激活虚拟环境
conda activate imagegenx && cd ~/ImageGenX

# 2. 启动Celery
celery -A app.worker worker --loglevel=info --concurrency=8

# 3. 验证启动（终端输出：celery@xxx ready 则正常）
```

#### 步骤9：启动监控服务（可选，Docker Compose一键启动）
##### 9.1 编写监控Docker Compose文件（`docker-compose-monitor.yaml`）
```yaml
version: '3.8'

services:
  # Prometheus（指标采集）
  prometheus:
    image: prom/prometheus:v2.40.0
    container_name: imagegenx-prometheus
    restart: always
    volumes:
      - ./configs/monitor/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - imagegenx-network

  # Grafana（可视化）
  grafana:
    image: grafana/grafana:9.0.0
    container_name: imagegenx-grafana
    restart: always
    volumes:
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123  # 初始密码
    networks:
      - imagegenx-network

  # Jaeger（链路追踪）
  jaeger:
    image: jaegertracing/all-in-one:1.42
    container_name: imagegenx-jaeger
    restart: always
    ports:
      - "16686:16686"  # 控制台端口
      - "14268:14268"  # 采集端口
    networks:
      - imagegenx-network

networks:
  imagegenx-network:
    driver: bridge

volumes:
  prometheus-data:
  grafana-data:
```

##### 9.2 启动监控服务
```bash
# 1. 项目根目录执行
docker-compose -f docker-compose-monitor.yaml up -d

# 2. 验证监控服务
# Grafana：浏览器访问 http://服务器IP:3000 ，用户名admin，密码admin123（首次登录需修改）
# Jaeger：浏览器访问 http://服务器IP:16686 ，能看到控制台则正常
```

### 第三章：功能测试（API调用→结果验证→任务管理）
#### 步骤1：API测试准备（获取API Key）
```bash
# 1. 初始化测试用户和API Key（执行脚本）
python scripts/create_test_user.py

# 2. 记录输出的API Key（后续调用接口需用）
# 输出示例：Test user created successfully! API Key: imagegenx_test_1234567890abcdef
```

#### 步骤2：调用Text2Img接口生成图片（3种方式，任选其一）
##### 方式1：curl命令（终端直接调用）
```bash
# 替换以下API_KEY为步骤1生成的实际Key
API_KEY="imagegenx_test_1234567890abcdef"

curl -X POST http://localhost:8000/api/v1/image/generate \
-H "Content-Type: application/json" \
-H "X-API-Key: $API_KEY" \
-d '{
  "text": "电商白底图，黑色运动鞋，高清细节，无阴影",
  "params": {
    "resolution": "512×512",
    "style": "photorealistic",
    "steps": 50,
    "cfg_scale": 7.5,
    "postprocess": {
      "super_resolution": "2x",
      "watermark": {
        "enable": false
      }
    }
  }
}'
```

##### 方式2：Postman调用（可视化操作）
1. 打开Postman，新建POST请求，URL填：`http://服务器IP:8000/api/v1/image/generate`
2. Headers添加：
   - Key：`Content-Type`，Value：`application/json`
   - Key：`X-API-Key`，Value：步骤1生成的API Key
3. Body选raw→JSON，粘贴以下内容：
```json
{
  "text": "电商白底图，黑色运动鞋，高清细节，无阴影",
  "params": {
    "resolution": "512×512",
    "style": "photorealistic",
    "steps": 50,
    "cfg_scale": 7.5,
    "postprocess": {
      "super_resolution": "2x",
      "watermark": {
        "enable": false
      }
    }
  }
}
```
4. 点击Send，查看响应结果

##### 方式3：Swagger UI调用（浏览器直接操作）
1. 浏览器访问：`http://服务器IP:8000/docs`
2. 找到`/api/v1/image/generate`接口，点击「Try it out」
3. 输入X-API-Key和请求体（同方式2），点击「Execute」
4. 查看响应结果

#### 步骤3：解析响应结果
成功响应示例：
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "task_id": "img202406011234567890123",
    "status": "processing",
    "progress": 30,
    "estimated_time": 1.5,
    "query_url": "http://localhost:8000/api/v1/image/task/img202406011234567890123",
    "temp_url": "http://localhost:9000/temp/img202406011234567890123_thumb.webp"
  }
}
```
- `task_id`：任务唯一ID，用于查询状态和结果
- `status`：任务状态（processing=处理中，success=成功，failed=失败）
- `query_url`：任务查询接口
- `temp_url`：临时缩略图URL（可直接浏览器访问预览）

#### 步骤4：查询任务状态与结果
```bash
# 替换TASK_ID为响应中的实际task_id
TASK_ID="img202406011234567890123"
API_KEY="imagegenx_test_1234567890abcdef"

curl -X GET "http://localhost:8000/api/v1/image/task/$TASK_ID" \
-H "X-API-Key: $API_KEY"
```

成功结果响应示例（status=success）：
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "task_id": "img202406011234567890123",
    "status": "success",
    "progress": 100,
    "result": {
      "original_url": "http://localhost:9000/imagegenx-bucket/20240601/img202406011234567890123_1024x1024.webp",  # 2x超分后1024×1024
      "thumb_url": "http://localhost:9000/imagegenx-bucket/20240601/img202406011234567890123_256x256.webp",
      "resolution": "1024×1024",
      "format": "webp",
      "size": 123456  # 图片大小（字节）
    },
    "create_time": "2024-06-01T12:34:56",
    "finish_time": "2024-06-01T12:34:57"
  }
}
```

#### 步骤5：下载生成的图片
1. 复制响应中的`original_url`，浏览器直接访问即可下载
2. 或用curl下载：
```bash
curl -O -L "http://localhost:9000/imagegenx-bucket/20240601/img202406011234567890123_1024x1024.webp"
```

#### 步骤6：批量生成图片测试
```bash
API_KEY="imagegenx_test_1234567890abcdef"

curl -X POST http://localhost:8000/api/v1/image/batch-generate \
-H "Content-Type: application/json" \
-H "X-API-Key: $API_KEY" \
-d '{
  "texts": [
    "电商白底图，红色T恤",
    "电商白底图，蓝色牛仔裤",
    "电商白底图，白色运动鞋"
  ],
  "params": {
    "resolution": "512×512",
    "style": "photorealistic",
    "batch_size": 3
  },
  "callback_url": "https://your-server.com/callback"
}'
```

响应会返回批量任务ID，可通过`/api/v1/image/batch-task/{batch_task_id}`查询所有子任务状态，完成后支持打包下载。

### 第四章：自定义扩展（手把手教接入ControlNet模型）
#### 步骤1：下载ControlNet模型（Canny版本）
```bash
# 1. 创建模型目录
mkdir -p models/controlnet/canny

# 2. 下载ControlNet Canny模型（Hugging Face官方链接）
wget -P models/controlnet/canny https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth

# 3. 验证模型下载完成
ls models/controlnet/canny/control_v11p_sd15_canny.pth  # 存在则正常
```

#### 步骤2：实现ControlNet模型接口
创建文件：`models/controlnet/canny/model.py`，复制以下代码：
```python
from app.model.base import ModelInterface
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import numpy as np
import cv2

class ControlNetCannyModel(ModelInterface):
    def __init__(self, config):
        self.config = config
        self.pipeline = self.load()
    
    def load(self):
        """加载ControlNet+Canny模型"""
        # 加载ControlNet Canny模型
        controlnet = ControlNetModel.from_pretrained(
            self.config["controlnet_path"],
            torch_dtype=torch.float16
        )
        # 加载基础SD 1.5模型
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config["base_model_path"],
            controlnet=controlnet,
            torch_dtype=torch.float16,
            device_map="auto"  # 自动分配GPU/CPU
        )
        # 启用TensorRT加速（已在配置中开启）
        if self.config.get("use_tensorrt", False):
            pipeline.to("cuda", torch.float16)
            pipeline.enable_tensorrt()
        return pipeline
    
    def preprocess_canny(self, image):
        """Canny边缘检测预处理（输入图片→控制图）"""
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)
    
    def infer(self, input_data):
        """推理逻辑：文本+原始图片+控制图→生成图片"""
        text = input_data["text"]  # 生成提示词
        image = input_data["image"]  # 原始图片（PIL.Image）
        params = input_data["params"]
        
        # 生成Canny控制图（若未传入控制图，自动处理）
        if "control_image" in input_data and input_data["control_image"] is not None:
            control_image = input_data["control_image"]
        else:
            control_image = self.preprocess_canny(image)
        
        # 执行推理
        generated_image = self.pipeline(
            prompt=text,
            image=control_image,
            num_inference_steps=params.get("steps", 50),
            guidance_scale=params.get("cfg_scale", 7.5),
            width=params.get("width", 512),
            height=params.get("height", 512)
        ).images[0]
        
        return generated_image
    
    def release(self):
        """释放资源"""
        del self.pipeline
        torch.cuda.empty_cache()
```

#### 步骤3：添加模型配置文件
创建文件：`configs/model/controlnet_canny.yaml`，复制以下内容：
```yaml
model_type: controlnet_canny
base_model_path: ./models/sd1.5  # 基础SD 1.5模型路径
controlnet_path: ./models/controlnet/canny  # ControlNet模型路径
use_tensorrt: true  # 启用TensorRT加速
inference_precision: fp16  # 推理精度
max_batch_size: 4  # ControlNet显存占用较高，批处理大小设小些
```

#### 步骤4：注册模型
修改文件：`app/model/registry.py`，添加以下代码（在文件末尾）：
```python
# 导入ControlNet Canny模型
from models.controlnet.canny.model import ControlNetCannyModel

# 注册模型（key为model_type，与配置文件一致）
model_registry["controlnet_canny"] = ControlNetCannyModel
```

#### 步骤5：重启服务，测试ControlNet接口
```bash
# 1. 停止之前的Go调度服务和Python业务服务
# 停止Go服务：ps aux | grep scheduler | grep -v grep | awk '{print $2}' | xargs kill
# 停止Python服务：Ctrl+C（前台运行）或 ps aux | grep uvicorn | grep -v grep | awk '{print $2}' | xargs kill

# 2. 重启Go调度服务
nohup go run cmd/scheduler/main.go > logs/scheduler.log 2>&1 &

# 3. 重启Python业务服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 4. 测试ControlNet接口（需上传原始图片和控制图，用Postman更方便）
```

Postman测试ControlNet接口：
1. 接口URL：`http://服务器IP:8000/api/v1/image/controlnet`
2. Headers：`Content-Type: application/json`、`X-API-Key: 你的API Key`
3. 请求体：
```json
{
  "text": "科技风建筑，蓝色调，高清",
  "image_url": "http://localhost:9000/imagegenx-bucket/20240601/input.jpg",  # 原始图片URL（需先上传到MinIO）
  "control_type": "canny",  # 控制类型
  "params": {
    "resolution": "512×512",
    "steps": 50,
    "cfg_scale": 7.5
  }
}
```
4. 点击Send，查看生成结果（生成的图片会贴合Canny控制图的边缘轮廓）

### 第五章：监控与运维（Grafana面板配置+常见问题排查）
#### 步骤1：Grafana面板配置（生图专属监控）
1. 浏览器访问Grafana：`http://服务器IP:3000`，登录（用户名admin，密码admin123）
2. 添加数据源：
   - 左侧菜单→Data Sources→Add data source→选择Prometheus
   - URL填：`http://prometheus:9090`（Docker内部网络，若外部访问填服务器IP:9090）
   - 点击Save & Test，显示Data source is working则正常
3. 导入生图监控面板：
   - 左侧菜单→Dashboards→Import→输入面板ID（项目提供的生图专属面板ID：12345，或上传`configs/monitor/grafana_dashboard.json`）
   - 选择数据源为Prometheus，点击Import
4. 查看监控面板：包含QPS、延迟、GPU利用率、任务成功率等核心指标

#### 步骤2：常见问题排查（按报错类型快速定位）
| 问题现象 | 排查步骤 | 解决办法 |
|----------|----------|----------|
| 调用API返回「API Key无效」 | 1. 检查X-API-Key是否正确；2. 检查脚本`create_test_user.py`是否执行成功 | 重新执行`python scripts/create_test_user.py`，使用新生成的API Key |
| 生图任务一直处于processing状态 | 1. 查看Celery日志（是否有报错）；2. 查看Go调度服务日志（logs/scheduler.log）；3. 查看GPU利用率（nvidia-smi） | 1. 重启Celery：`pkill -f celery && celery -A app.worker worker --loglevel=info`；2. 若GPU利用率100%，扩容GPU或降低并发 |
| 推理失败，日志显示「显存不足」 | 1. 查看生成分辨率和Batch Size；2. 查看GPU显存占用（nvidia-smi） | 1. 降低分辨率（如1024→512）；2. 减小max_batch_size（application.yaml中）；3. 启用INT8量化（需校准） |
| MinIO访问失败，返回「Access Denied」 | 1. 检查application.yaml中的minio.access_key和secret_key；2. 检查MinIO桶是否存在 | 1. 确保与docker-compose-mid.yaml中的MINIO_ROOT_USER/Password一致；2. 手动创建桶：访问MinIO控制台（http://服务器IP:9001），登录后创建bucket：imagegenx-bucket |
| 服务启动后，浏览器无法访问8000端口 | 1. 检查防火墙是否开放8000端口；2. 检查服务绑定的IP是否为0.0.0.0 | 1. 开放端口：`sudo ufw allow 8000`；2. 启动命令确保`--host 0.0.0.0` |

---

## 四、其余模块（保持原结构，补充教程相关说明）
### （一）项目概述、核心特性、技术架构（保持原内容）
### （二）环境要求（与教程步骤1一致，无需修改）
### （三）扩展指南（教程第四章已覆盖自定义模型接入，补充其他扩展场景的手把手步骤）
### （四）贡献指南（保持原内容）
### （五）许可证（保持原内容）
### （六）FAQ（补充教程中未覆盖的高频问题）

---

**声明**：本项目仅提供 AIGC 生图片技术的工程化落地工具，不涉及模型训练本身。用户需自行确保所使用的模型及生成内容符合法律法规和伦理规范，不得用于侵权、虚假宣传、色情暴力等非法用途。

**教程更新说明**：若后续项目代码更新，教程将同步迭代，确保步骤可复现。如有问题，可通过GitHub Issues反馈：https://github.com/ImageGenX/ImageGenX/issues