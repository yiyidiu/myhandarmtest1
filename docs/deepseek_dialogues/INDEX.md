# DeepSeek 顾问记录索引

日期：2026-08-14

## 当前状态

`DEEPSEEK_HARNESS_CONNECTED_WITH_USER_AUTHORIZATION`

初次探测时只确认了 `127.0.0.1:3080` 的 DeepSeek Harness Web/RPC，实际模型适配器为
公网 `deepseek-official`，所以在用户尚未明确授权前没有发送提示。用户随后明确要求在
DeepSeek Harness 中实时看到 Codex 与 DeepSeek 的对话；此后 Codex 才建立会话并调用：

- Harness 会话：`Codex顾问对话：连续避障与真实物理抓取`
- Provider/model：`deepseek-official/deepseek-v4-pro`
- Reasoning effort：`max`
- DeepSeek 角色：只读顾问
- Codex 角色：唯一工作区写入者、测试者和最终验收者

打开 <http://127.0.0.1:3080> 并选择上述会话，可实时查看后续问答。

## 安全边界

- 顾问提示明确禁止工具、Shell、文件读写和子 Agent。
- `tools/deepseek_consult.py` 检测到任何 `tool/*` 事件会将咨询判为越界。
- 不记录或打印 API 密钥；配置仅保存 loopback Harness 地址、模型名和会话 ID。
- DeepSeek 回答只是建议，不能替代 Codex 的编译、单测和 Gazebo 实测。

## 记录位置

- `results/deepseek_dialogues/AUTOMATION_STATUS.json`：连接与授权事实。
- 本目录的时间戳 Markdown：公开提示、公开回答、Codex 审查和最终决定。
- `results/deepseek_dialogues/` 的同名 JSON：机器可读原始公开对话记录。
