# 命令参考

以下为常用命令分组速查。

## 初始化与会话

| 命令 | 说明 |
|---|---|
| `nullclaw onboard --api-key sk-... --provider openrouter` | 快速初始化 provider 与 key |
| `nullclaw onboard --interactive` | 交互式完整初始化 |
| `nullclaw onboard --channels-only` | 仅重配渠道与 allowlist |
| `nullclaw agent -m "..."` | 单条消息模式 |
| `nullclaw agent` | 交互模式 |

## 网关与服务

| 命令 | 说明 |
|---|---|
| `nullclaw gateway` | 启动长期运行 runtime（默认 `127.0.0.1:3000`） |
| `nullclaw gateway --port 8080` | 指定端口启动网关 |
| `nullclaw service install` | 安装后台服务 |
| `nullclaw service start` | 启动后台服务 |
| `nullclaw service stop` | 停止后台服务 |
| `nullclaw service status` | 查看后台服务状态 |
| `nullclaw service uninstall` | 卸载后台服务 |

## 诊断与状态

| 命令 | 说明 |
|---|---|
| `nullclaw doctor` | 系统健康诊断 |
| `nullclaw status` | 全局状态总览 |
| `nullclaw channel status` | 渠道健康状态 |

## 渠道与扩展

| 命令 | 说明 |
|---|---|
| `nullclaw channel start telegram` | 启动指定渠道 |
| `nullclaw channel start discord` | 启动指定渠道 |
| `nullclaw channel start signal` | 启动指定渠道 |
| `nullclaw skills list` | 列出技能包 |
| `nullclaw skills install <name>` | 安装技能包 |
| `nullclaw skills remove <name>` | 移除技能包 |
| `nullclaw skills info <name>` | 查看技能详情 |

## 定时任务与迁移

| 命令 | 说明 |
|---|---|
| `nullclaw cron list` | 查看任务列表 |
| `nullclaw cron add ...` | 添加 cron 任务 |
| `nullclaw cron once ...` | 执行一次性任务 |
| `nullclaw cron runs` | 查看执行记录 |
| `nullclaw migrate openclaw --dry-run` | 预演迁移 OpenClaw |
| `nullclaw migrate openclaw` | 执行迁移 |

## 硬件与模型

| 命令 | 说明 |
|---|---|
| `nullclaw hardware scan` | 扫描硬件设备 |
| `nullclaw hardware flash` | 烧录设备 |
| `nullclaw hardware monitor` | 监控硬件 |
| `nullclaw models list` | 模型目录 |
| `nullclaw models info` | 模型信息 |
| `nullclaw models benchmark` | 模型基准测试 |
