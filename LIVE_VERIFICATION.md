# 真实服务验证

日期：2026-11-21。DeepSeek deepseek-flash，使用完全虚构的演示甲/演示乙档案，不将原始档案用于本轮云端验证。

| 测试 | 实际结果 |
|---|---|
| POST /v1/chat/completions 同步 | HTTP200，A型血[1]，guard=ok |
| 同接口 stream=true | HTTP200，相同正确引用，收到[DONE] |
| 未记载的银行账号 | HTTP200，资料中未提及该信息，guard=abstain |

本轮使用BM25以单独验证模型与HTTP协议；真实混检和重排另有本地评测及烟测记录。原始响应在忽略目录private-evidence/。少量烟测不能替代批量答案准确率和人工引用支持评测。
