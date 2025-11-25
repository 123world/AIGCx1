# VideoGenX: 企业级AIGC生视频智能生成开源平台
![GitHub License](https://img.shields.io/github/license/VideoGenX/VideoGenX)
![GitHub Stars](https://img.shields.io/github/stars/VideoGenX/VideoGenX)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Go Version](https://img.shields.io/badge/go-1.20%2B-blue)
![GPU Required](https://img.shields.io/badge/GPU-NVIDIA%20CUDA%2011.8%2B-red)

## 一、项目概述
### 1. 项目目标
VideoGenX 是一款**高可用、高并发、低延迟**的企业级AIGC生视频开源平台，支持通过文本描述、图片输入快速生成营销视频、产品展示视频等商业化内容，核心定位为「算法工程化落地工具+企业级服务架构模板」，可直接适配电商、广告、传媒等行业的视频自动化生成需求。

### 2. 核心价值
- 解决AIGC生视频模型工程化落地痛点：提供从模型转换、推理加速到服务部署的全链路解决方案
- 支持高并发场景：通过「Python+Go混合架构」「动态批处理」「负载均衡」支撑日均10万+生成任务
- 降低企业接入成本：提供标准化API、可配置化生成参数、完善的监控运维体系，开箱即用
- 高度可扩展：支持自定义模型接入、存储方案替换、功能模块插件化扩展

### 3. 应用场景
- 电商行业：产品图片→自动生成促销视频、商品展示视频（支持添加字幕、背景音乐）
- 广告行业：文本描述→自动生成品牌广告视频（支持自定义风格：科技风、文艺风等）
- 传媒行业：脚本文本→自动生成短视频片段（支持多镜头拼接、转场效果）
- 企业服务：企业Logo+产品信息→自动生成企业宣传视频

## 二、核心特性
### 1. 多模型兼容与推理加速
- 支持主流生视频模型：Sora同类模型（如Pika Labs、Runway Gen-2）、Stable Video Diffusion等
- 支持生图模型扩展：Stable Diffusion、MidJourney风格迁移（可作为视频帧生成基础）
- 推理加速方案：TensorRT/ONNX Runtime量化（FP16/INT8）、算子融合、模型并行/张量并行
- 显存优化：动态批处理、模型分片加载、中间张量自动释放，8GB GPU可支持1080P视频生成

### 2. 高并发服务架构
- 混合架构：Python（FastAPI）负责业务层高并发IO，Go负责底层任务调度与资源管理
- 任务调度：支持优先级调度（付费用户/普通用户分级）、任务排队、重试、限流熔断
- 弹性扩容：适配K8s HPA，基于QPS、GPU利用率自动扩缩容
- 跨语言通信：GRPC实现Python与Go模块高效通信，支持双向流

### 3. 灵活的存储方案
- 分层存储：PostgreSQL（任务元数据）、Redis（热点缓存）、对象存储（MinIO/S3/OSS，视频文件）
- 数据优化：视频H.265编码压缩、图片WebP格式转换、大文件分片上传/下载
- 安全访问：生成文件签名URL、访问权限控制、过期自动清理

### 4. 全链路可观测性
- 监控体系：Prometheus+Grafana（业务指标、资源指标、GPU指标可视化）
- 日志分析：结构化日志（JSON格式）、ELK/Loki集成支持
- 链路追踪：Jaeger全链路耗时追踪（请求→调度→推理→存储）
- 告警机制：多级别告警（钉钉/邮件/短信）、故障自动上报

### 5. 功能完整性
- 生成参数可配置：分辨率（720P/1080P/4K）、帧率（24/30fps）、时长（15s-60s）、风格、字幕、背景音乐
- 视频后处理：镜头拼接、转场效果、水印添加、格式转码（MP4/AVI/WebM）
- 任务管理：同步查询结果、异步回调通知、任务状态查询（通过任务ID）
- 权限控制：API密钥认证、用户级限流、功能权限分级

## 三、技术架构
### 1. 整体架构图（分层设计）
```
┌─────────────────────────────────────────────────────────────────┐
│ 接入层                     Nginx + API Gateway                   │
│ （限流、认证、请求转发、CDN加速）                                 │
├─────────────────────────────────────────────────────────────────┤
│ 业务层               Python + FastAPI + Celery                   │
│ （任务接收、参数校验、业务逻辑、结果回调）                         │
├─────────────────────────────────────────────────────────────────┤
│ 调度层                   Go + GRPC + 调度算法                    │
│ （GPU资源监控、任务分发、负载均衡、优先级管理）                   │
├─────────────────────────────────────────────────────────────────┤
│ 推理层           TensorRT + ONNX Runtime + Diffusers            │
│ （模型加载、推理加速、视频帧生成、后处理）                       │
├─────────────────────────────────────────────────────────────────┤
│ 存储层       PostgreSQL + Redis + MinIO/S3/OSS                   │
│ （元数据存储、缓存、大文件存储）                                 │
├─────────────────────────────────────────────────────────────────┤
│ 监控层       Prometheus + Grafana + Jaeger + ELK/Loki           │
│ （指标采集、可视化、链路追踪、日志分析）                         │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 核心模块说明
| 模块名称         | 技术栈                          | 核心职责                                                                 |
|------------------|---------------------------------|--------------------------------------------------------------------------|
| 任务接入模块     | FastAPI + Pydantic              | 提供RESTful API，参数校验（生成参数合法性）、API认证、任务ID生成          |
| 任务调度模块     | Go + GRPC + 自定义调度算法      | GPU负载监控、任务优先级排序、跨GPU任务分发、动态批处理调度                |
| 模型推理模块     | Python + TensorRT + Diffusers   | 模型格式转换、推理加速、视频帧生成、后处理（拼接/转码/字幕添加）          |
| 存储管理模块     | SQLAlchemy + Redis + MinIO SDK  | 元数据CRUD、热点缓存管理、大文件存储、签名URL生成                        |
| 监控告警模块     | Prometheus Client + Grafana API | 指标采集（QPS/延迟/成功率/GPU使用率）、告警规则配置、故障通知            |
| 跨语言通信模块   | GRPC + Protobuf                 | Python与Go模块数据传输、服务注册与发现、通信超时处理                    |

## 四、环境要求
### 1. 开发环境
| 依赖类型         | 版本要求                          | 说明                                                                 |
|------------------|-----------------------------------|----------------------------------------------------------------------|
| 操作系统         | Linux（Ubuntu 20.04+/CentOS 7+）  | 推荐Linux，Windows需适配WSL2，MacOS不支持GPU推理                      |
| Python           | 3.9-3.11                          | 核心业务层开发语言                                                   |
| Go               | 1.20+                             | 调度层开发语言                                                       |
| GPU              | NVIDIA GPU（显存≥8GB）            | 支持CUDA 11.8+，推荐A10、A100（高并发场景）                          |
| CUDA             | 11.8+                             | 模型推理依赖                                                         |
| TensorRT         | 8.6+                              | GPU推理加速                                                         |
| ONNX Runtime     | 1.15+                             | 跨硬件推理支持                                                       |
| 中间件           | Redis 6.2+、PostgreSQL 14+、MinIO | 缓存、元数据存储、文件存储（可替换为S3/OSS）                          |
| 监控工具         | Prometheus 2.40+、Grafana 9.0+    | 指标监控与可视化                                                     |
| 容器化工具       | Docker 20.10+、Docker Compose 2.10+ | 开发环境快速部署                                                     |

### 2. 生产环境额外要求
- K8s集群：1.24+（支持GPU调度、HPA弹性扩容）
- 网络：支持GRPC、HTTP/2，建议带宽≥100Mbps（高并发文件传输）
- 存储：对象存储支持分片上传、CDN加速（用户访问生成视频）
- 高可用：多节点部署（避免单点故障）、数据备份策略（PostgreSQL主从复制）

## 五、快速开始
### 1. 环境准备
- 安装依赖：执行`requirements.txt`（Python依赖）和`go.mod`（Go依赖）
- 配置GPU环境：安装CUDA、TensorRT，验证`nvidia-smi`和`trtexec`可正常执行
- 启动中间件：通过Docker Compose启动Redis、PostgreSQL、MinIO（参考`docker-compose.yml`）

### 2. 配置文件修改
- 核心配置文件：`configs/application.yaml`（支持YAML/JSON格式）
  - 数据库配置：PostgreSQL/Redis连接信息
  - 存储配置：对象存储类型（MinIO/S3/OSS）、访问密钥、存储路径
  - GPU配置：GPU数量、并发限制、推理精度（FP16/INT8）
  - 服务配置：API端口、GRPC通信端口、限流阈值、任务优先级规则

### 3. 服务启动
```bash
# 1. 启动Go调度服务（后台运行）
cd cmd/scheduler && go run main.go

# 2. 启动Python业务服务（支持热重载）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 3. 启动Celery任务队列（处理异步任务）
celery -A app.worker worker --loglevel=info

# 4. 启动监控服务（可选）
docker-compose -f docker-compose-monitor.yml up -d
```

### 4. 功能测试
- 调用API生成视频：通过Postman/ curl调用`/api/v1/video/generate`接口，传入文本描述、生成参数
- 查询任务状态：通过`/api/v1/video/task/{task_id}`查询生成进度
- 访问生成结果：通过接口返回的签名URL下载视频文件
- 查看监控面板：访问Grafana（默认端口3000），查看QPS、延迟、GPU利用率等指标

## 六、部署方案
### 1. 开发环境部署（Docker Compose）
- 一键部署：`docker-compose up -d`，自动启动所有依赖服务（中间件、业务服务、调度服务）
- 数据持久化：配置本地目录挂载，保障PostgreSQL/Redis/MinIO数据不丢失
- 端口映射：API端口8000、Grafana端口3000、MinIO端口9000

### 2. 生产环境部署（K8s）
- 资源配置：提供`k8s/`目录下的Deployment、Service、HPA、ConfigMap配置文件
- GPU调度：通过NVIDIA GPU Operator配置GPU资源调度，支持MPS GPU共享
- 弹性扩容：基于QPS（HPA指标：`http_requests_total`）和GPU利用率（`nvidia_gpu_utilization`）自动扩缩容
- 滚动更新：支持服务无停机更新，保障业务连续性

### 3. 集群部署注意事项
- 多节点部署：业务服务、调度服务、推理服务可分布式部署，通过K8s Service实现服务发现
- 存储选型：生产环境推荐使用云厂商对象存储（S3/OSS），稳定性更高
- 安全配置：API密钥通过K8s Secret管理，禁止明文配置；开启HTTPS加密传输
- 备份策略：PostgreSQL每日自动备份，对象存储文件开启版本控制

## 七、扩展指南
### 1. 接入自定义生视频模型
1. 在`models/`目录下创建模型适配目录（如`models/custom_video_model/`）
2. 实现`ModelInterface`接口（定义`load()`、`infer()`、`release()`方法）
3. 在配置文件中指定模型类型为自定义模型，配置模型路径、推理参数
4. 重启服务，自动加载新模型

### 2. 替换存储方案
- 替换对象存储：修改`storage/object_storage/`目录下的适配器，实现`ObjectStorageInterface`（支持S3/OSS/MinIO）
- 替换缓存：修改`storage/cache/`目录下的Redis适配器，支持替换为Memcached等
- 替换数据库：修改`storage/database/`目录下的PostgreSQL适配器，支持替换为MySQL等

### 3. 扩展功能模块
- 新增视频后处理功能：在`processor/video_postprocess/`目录下添加插件（如AI字幕生成、智能背景音乐匹配）
- 新增模型加速方案：在`inference/accelerator/`目录下实现新的推理加速适配器（如vLLM）
- 新增监控指标：在`monitor/metrics/`目录下添加自定义指标采集逻辑

## 八、贡献指南
### 1. 贡献流程
1. Fork本仓库到个人账号
2. 创建特性分支：`git checkout -b feature/xxx`（功能开发）/`bugfix/xxx`（bug修复）
3. 提交代码：遵循代码规范（Python使用black格式化，Go使用gofmt格式化）
4. 编写测试：为新增功能添加单元测试（`tests/`目录下）
5. 提交PR：描述功能/修复内容、测试结果，关联相关Issue

### 2. 代码规范
- Python：遵循PEP 8规范，使用black进行代码格式化（`black --line-length 120 .`）
- Go：遵循Go官方规范，使用gofmt格式化代码（`gofmt -w .`）
- 注释要求：核心函数、类需添加文档字符串，说明功能、参数、返回值
- 日志规范：使用结构化日志，包含`task_id`、`user_id`、`level`、`message`等字段

### 3. Issue提交规范
- 功能需求：标题前缀`[Feature]`，描述需求场景、预期效果
- Bug报告：标题前缀`[Bug]`，描述复现步骤、错误日志、环境信息
- 优化建议：标题前缀`[Optimize]`，描述当前问题、优化方案、预期收益

## 九、许可证
本项目采用 **Apache License 2.0** 许可证，允许商业使用、修改、分发，但需保留原作者版权声明和许可证文件。使用本项目时，需遵守相关开源协议和AIGC内容生成的法律法规，不得用于非法用途。

## 十、FAQ
### 1.  GPU显存不足怎么办？
- 降低推理精度：配置文件中设置`inference_precision: FP16`（默认）或`INT8`
- 减小批处理大小：调整`batch_size: 1`（最低支持1）
- 降低生成分辨率：从1080P降至720P
- 启用模型分片：配置`model_parallel: true`，拆分模型到多个GPU

### 2. 高并发下任务超时如何处理？
- 检查GPU利用率：若达100%，需扩容GPU节点或启用动态批处理
- 优化任务队列：调整Celery Worker数量（建议为GPU核心数的2倍）
- 启用异步回调：优先使用异步接口（`callback_url`参数），避免同步阻塞
- 扩容队列：新增Redis节点，提升任务队列吞吐量

### 3. 如何对接自有业务系统？
- 调用标准化API：参考`docs/api文档.md`，支持RESTful API和GRPC两种调用方式
- 配置回调通知：生成完成后自动回调业务系统接口，传递任务状态和结果URL
- 集成权限系统：支持OAuth2.0认证，可对接企业自有权限系统

### 4. 生成视频效果不符合预期怎么办？
- 优化输入参数：调整`style`（风格）、`cfg_scale`（相关性）、`steps`（生成步数）
- 更换模型：配置文件中切换为其他生视频模型（如`model_type: pika_labs`）
- 增加输入约束：提供参考图片（`reference_image`参数），提升生成效果相关性

## 十一、联系与支持
- GitHub Issues：https://github.com/VideoGenX/VideoGenX/issues（优先反馈问题）
- 技术交流群：添加微信`VideoGenX_Admin`，备注「开源交流」加入社群
- 商业支持：提供企业级定制开发、私有化部署、技术培训服务，联系邮箱`contact@videogenx.com`

---

**声明**：本项目仅提供AIGC生视频技术的工程化落地工具，不涉及模型训练本身。用户需自行确保所使用的模型及生成内容符合法律法规和伦理规范，不得用于侵权、虚假宣传等非法用途。