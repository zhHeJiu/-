# Moral Dilemma Assessor

AI 驱动的客观冲突分析系统。项目通过 5 步工作流和多智能体陪审团机制，对人际冲突或道德困境进行结构化分析，并给出双方错误度评分与调解建议。

## 功能特点

- 输入质量检测，事实不足时自动追问
- 法庭证词级中立化转译
- 正方/反方多视角重构
- 5 个异构 AI 裁判并发评分
- 加权聚合双方错误度
- SSE 流式进度与 token 消耗展示
- 网页可视化修改大模型/API 配置，并自动保存到本地配置文件

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python + FastAPI |
| 前端 | 原生 HTML/JS + TailwindCSS |
| LLM | MiniMax-M2.7-highspeed 或兼容 OpenAI Chat Completions 的模型 |
| API | MiniMax OpenAI 兼容接口 |

## 快速启动

Windows 用户可以双击：

```text
一键启动.bat
```

也可以手动启动：

```bash
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8011
```

然后打开：

```text
frontend/index.html
```

## 发布形态

当前发布包是源码版 Web 项目，不包含 EXE 打包产物。

目录里已经包含前端、后端、一键启动脚本和示例配置。Windows 用户优先双击 `一键启动.bat` 使用；如果需要部署到服务器，可以按上面的手动启动命令运行 FastAPI 后端，再打开 `frontend/index.html`。

## 模型配置

首次打开网页后，在右侧「模型配置」面板填写：

- API Key
- Base URL
- Model
- 前置温度
- 评分温度
- 重试次数

配置会自动保存到本地 `app_config.json`。该文件可能包含密钥，已被 `.gitignore` 忽略，不应提交到 GitHub。

## License

This project is source-available under the PolyForm Noncommercial License 1.0.0.

Commercial use is not permitted without separate prior written permission from the author.

This is not an OSI-approved open-source license. It is intended for public source sharing while the project is still unfinished and commercial plans remain undecided.

Required Notice: Moral Dilemma Assessor is source-available for noncommercial use only. Commercial use requires separate prior written permission from zh_zh.
