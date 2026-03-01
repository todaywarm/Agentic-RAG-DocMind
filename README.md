# DocMind - 企业级 RAG 智能问答系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Custom-green.svg" alt="Framework">
  <img src="https://img.shields.io/badge/Embedding-BGE--M3-orange.svg" alt="Embedding">
  <img src="https://img.shields.io/badge/Reranker-BGE--Reranker--v2--m3-red.svg" alt="Reranker">
</p>

## 🌟 项目亮点

- **三路混合检索**：HyDE + 语义重写 + BM25，Recall@3 达 93%+
- **BGE SOTA 模型**：BGE-M3 Embedding + BGE-Reranker-v2-m3
- **三级置信度过滤**：有效抑制幻觉，提高答案可信度
- **完整评估体系**：Recall/MRR/NDCG + 消融实验
- **模块化设计**：清晰的代码架构，易于扩展和维护

## 🏗️ 项目结构

```
DocMind/
├── docmind/                    # 核心代码
│   ├── core/                   # 核心基础设施
│   │   ├── config.py           # 全局配置
│   │   └── exceptions.py       # 自定义异常
│   │
│   ├── llm/                    # LLM 模块
│   │   ├── client.py           # LLM 客户端
│   │   └── prompts.py          # Prompt 模板
│   │
│   ├── embedding/              # Embedding 模块
│   │   ├── base.py             # 基类
│   │   ├── local_bge.py        # 本地 BGE-M3
│   │   └── remote_bge.py       # 远程 API
│   │
│   ├── reranker/               # Reranker 模块
│   │   ├── base.py             # 基类
│   │   ├── local_bge.py        # 本地 BGE-Reranker
│   │   └── remote_bge.py       # 远程 API
│   │
│   ├── retrieval/              # 检索模块
│   │   ├── vector_store.py     # 向量存储 (Qdrant)
│   │   ├── bm25_store.py       # BM25 存储
│   │   ├── query_processor.py  # 查询处理
│   │   └── hybrid_retriever.py # 混合检索器
│   │
│   ├── document/               # 文档处理
│   │   ├── loader.py           # 文档加载器
│   │   └── processor.py        # 文档处理器
│   │
│   ├── generation/             # 生成模块
│   │   └── answer_generator.py # 答案生成器
│   │
│   └── agent/                  # Agent 模块
│       └── rag_agent.py        # RAG 智能体
│
├── evaluation/                 # 评估系统
│   ├── retrieval_metrics.py    # 检索指标
│   ├── generation_metrics.py   # 生成指标
│   ├── ablation_study.py       # 消融实验
│   └── run_evaluation.py       # 评估入口
│
├── data/                       # 数据目录
│   ├── raw/                    # 原始文档
│   └── evaluation/             # 评估数据
│
├── cli.py                      # CLI 入口
├── requirements.txt            # 依赖
└── README.md                   # 文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd DocMind
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```bash
# LLM 配置 (必填)
LLM_API_KEY=your_deepseek_api_key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL_ID=deepseek-chat

# RAG API 配置 (可选，使用远程 Embedding/Reranker)
RAG_API_KEY=your_siliconflow_api_key
```

### 3. 启动 CLI

```bash
python cli.py
```

### 4. 添加文档

```bash
# 命令行添加
python cli.py --add /path/to/document.pdf

# 或在交互模式中
/add /path/to/document.pdf
```

## 📊 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Query Processor                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Context Rewrite│  │ Query Rewrite│  │    HyDE     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Hybrid Retriever                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Vector(HyDE) │  │Vector(Rewrite)│  │    BM25     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                              │                              │
│                              ▼                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │              Merge & Deduplicate                 │      │
│  └──────────────────────────────────────────────────┘      │
│                              │                              │
│                              ▼                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │              BGE Reranker                        │      │
│  └──────────────────────────────────────────────────┘      │
│                              │                              │
│                              ▼                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │         Three-Tier Confidence Filter             │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Answer Generator                          │
│  ┌──────────────────────────────────────────────────┐      │
│  │                 DeepSeek LLM                     │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Response                             │
│           (Answer + Sources + Confidence)                   │
└─────────────────────────────────────────────────────────────┘
```

## 📈 评估结果

### 检索指标

| 指标 | Baseline | + BM25 | + Rewrite | + HyDE | Full Pipeline |
|------|----------|--------|-----------|--------|---------------|
| Recall@1 | 33.33% | 40.00% | 46.67% | 73.33% | **86.67%** |
| Recall@3 | 60.00% | 80.00% | 80.00% | 93.33% | **93.33%** |
| MRR | 0.4611 | 0.5856 | 0.6300 | 0.8133 | **0.8500** |

### 各模块增益

- **BM25 混合检索**: Recall@3 +20%
- **Query Rewrite**: MRR +4.4%
- **HyDE**: Recall@3 +13.3%, MRR +18.3%
- **Reranker**: 排序质量显著提升

## 🛠️ CLI 命令

```bash
# 交互模式
python cli.py

# 添加文档
python cli.py --add <file_path>

# 重置知识库
python cli.py --reset

# 运行评估
python cli.py --eval
```

### 交互模式命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助 |
| `/add <file>` | 添加文档 |
| `/reset` | 重置知识库 |
| `/clear` | 清除对话历史 |
| `/logs` | 显示检索日志 |
| `/quit` | 退出 |

## 📝 开发日志

- **v1.0**: 基础 RAG 流程
- **v1.5**: 混合检索 + BGE Reranker
- **v2.0**: 模块化重构，完整评估体系

## 📄 License

MIT License
