# health-rag

本地健康档案 RAG 学习项目，保留 OpenAI 兼容 /v1/chat/completions 接口，支持普通响应和经过同一引用守卫的整段缓冲 SSE。请仅监听 127.0.0.1；本项目无鉴权，不用于真实医疗决策。

## 启动
```powershell
.venv/Scripts/python -m pip install -r requirements-local.txt
# 填写 .env 的 LLM_API_KEY；DeepSeek 地址 https://api.deepseek.com，模型 deepseek-flash
.venv/Scripts/python scripts/demo_index.py --with-vectors
.venv/Scripts/python -m uvicorn app.api:app --host 127.0.0.1 --port 8012
```
浏览器访问 http://127.0.0.1:8012。BGE-M3 和 BGE reranker 必须预先下载至 HF_HOME；默认 local_files_only 避免启动时隐式下载。只装 requirements.txt 可运行 BM25，语义重排不可用时 trace 明确显示 lexical 降级。配置错误会报错，环境变量优先于 .env。索引构建后需要重启服务。

## 评测证据
原始代码在新环境运行 17 个测试通过。原始 BM25 32 个可回答问题，过滤开启 P@1 0.7812、旧 Recall@5 1.0，关闭 P@1 0.7188、旧 Recall@5 1.0，见 eval/baseline_original.json。旧指标实际是至少命中一个证据的 Hit@k，且旧判分未要求患者匹配，不能与改进后的指标直接比较。改进评测要求患者匹配，并计算候选任意位置串档；未做真实 LLM 答案准确率评测，不将检索命中当作答案准确率。不可回答样本数量单独报告，答案质量尚未测量。

数据目录、密钥、日志及虚拟环境已忽略。现有样例来源及发布授权未核实，不纳入 Git。学习路径、练习与简历事实表见 学习指南.md。

冷启动可先 `python -m venv .venv`。不下载模型的演示先安装 requirements.txt，再执行 `.venv/Scripts/python scripts/demo_index.py`。此脚本生成完全虚构的演示甲/演示乙档案。24 项测试使用完全虚构的隔离索引，无需原始 PDF 或私有 data/。真实本地 BGE-M3 编码与 FAISS 混合检索烟测已通过：2 条完全虚构档案，hybrid 实际调用 bm25 与 vector，正确召回 demo-a（A 型），无降级，证据见 eval/local-vector-smoke.json。该烟测仅验证接通，不代表检索质量；BGE cross encoder 重排已完成原始样例真实推理评测，见下文。

复现环境见 requirements-lock.txt；本机启动执行 ./start.ps1，端口 8012。最新 BM25 报告 eval/current.json：32 个可回答问题，过滤 P@1 0.7812、Hit@5 1.0、MRR@5 0.878、任意候选串档率 0；无过滤 P@1 0.7188、Hit@5 1.0、MRR@5 0.833、任意候选串档率 0.9062。eval/current.json 对应的历史运行无向量索引，hybrid 实际仅使用 BM25，不能作为混合检索效果；随后真实双路烟测见 eval/local-vector-smoke.json，勿混用两次实验结论。每次索引更新先原子替换向量、再替换 chunks 和 meta，读取时验证向量与 chunks 双重哈希，不匹配时禁用向量。多文件更新不保证整体事务，进程间并发重建不受支持。

公开评测：先 `python scripts/demo_index.py`，再 `python -m app.cli eval --modes bm25 --questions eval/demo_questions.jsonl --out eval/demo_report.txt`。公开问题完全虚构，共 8 个可回答、1 个不可回答；后者仅计数，尚未评测生成拒答质量。原始本地评测需显式指定 `--questions eval/questions.jsonl`。

来源说明：NanGePlus/RagLangChainTest 仅作为外部参考，不属于用户原创成果；本项目是在用户提供的现有 health-rag 代码上增强。简历应区分已有实现、参考学习和本次新增工作。

真实 BGE 原始样例检索评测已完成，独立索引 data/bge-original，30 chunks；38 问中 32 可回答参与检索打分，6 不可回答仅计数。见 eval/bge-original.json：患者过滤开启时 vector P@1=.75、Hit@5=.9062、MRR=.8187，hybrid P@1=.8125、Hit@5=1、MRR=.8734；两者串档率均为0。关闭过滤时 hybrid P@1=.8438，但任意候选串档率达 .875，不能作为更安全答案质量提升。实际 trace 均包含 vector、无降级；本轮没有重排及 LLM 答案质量评测。原始默认索引保留。

真实 BGE reranker 评测 eval/bge-rerank-original.json：原始 38 问中 32 个可回答问题参与打分。BM25+CrossEncoder，过滤开启 P@1=.9688、Hit@5=1、MRR@5=0.975、串档率0；关闭过滤 P@1=.9688、Hit@5=1、MRR@5=0.9792、任意候选串档率 .9062。所有 trace.rerank=cross-encoder，无 lexical 降级；无 LLM，不能宣称生成答案质量。默认两个 BAAI 模型加载时固定官方快照 revision，冷缓存仅有固定 revision 快照时也可离线加载；自定义模型路径不强加 revision。

独立模型下载：`.venv/Scripts/python scripts/download_models.py` 自动读取 .env，固定快照及权重校验信息见 MODEL_MANIFEST.json。完整混检+重排烟测结果见 eval/full-pipeline-smoke.json。当前本机 .env 指向虚构档案 data/bge-demo；原始样例索引 data/bge-original 保留；公开克隆按上面的虚构数据初始化步骤运行。

真实 DeepSeek HTTP 烟测已通过：同步和SSE均回答虚构演示甲为A型并引用[1]；银行账号问题拒答，SSE发送[DONE]。仅为少量烟测，不作为原始38题答案正确率。
