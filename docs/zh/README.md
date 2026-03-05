# NullClaw 中文文档

本目录提供 NullClaw 的中文使用文档，覆盖安装、配置、运行与排障。

## 文档导航

- [安装指南](./installation.md)
- [配置指南](./configuration.md)
- [使用与运维](./usage.md)
- [架构总览](./architecture.md)
- [安全机制](./security.md)
- [Gateway API](./gateway-api.md)
- [命令参考](./commands.md)
- [开发指南](./development.md)

## 先看这 3 条

1. NullClaw 当前要求 **Zig 0.15.2**（精确版本）。
2. 默认配置文件路径为 `~/.nullclaw/config.json`（由 `nullclaw onboard` 生成）。
3. 首次上手建议先跑 `onboard --interactive`，再用 `agent` 和 `gateway` 验证。

## 最短上手路径（3 分钟）

```bash
brew install nullclaw
nullclaw onboard --interactive
nullclaw agent -m "你好，nullclaw"
```

如果你不用 Homebrew，请按 [安装指南](./installation.md) 走源码构建流程。
