# 开发指南

本项目开发与测试固定在 **Zig 0.15.2**。

## 本地构建与测试

```bash
zig build
zig build -Doptimize=ReleaseSmall
zig build test --summary all
```

建议在提交前至少执行：

```bash
zig build test --summary all
```

## 常用构建参数

```bash
zig build -Dchannels=telegram,cli
zig build -Dengines=base,sqlite
zig build -Dtarget=x86_64-linux-musl
zig build -Dversion=2026.3.1
```

说明：

- `channels`：裁剪编译进制中的渠道实现。
- `engines`：裁剪 memory engine。
- `target`：交叉编译目标。
- `version`：覆盖版本字符串（CalVer）。

## 代码结构（高频目录）

| 路径 | 说明 |
|---|---|
| `src/main.zig` | CLI 命令路由 |
| `src/config.zig` | 配置加载与环境覆盖 |
| `src/gateway.zig` | 网关与 webhook |
| `src/security/` | 安全与沙箱 |
| `src/providers/` | 模型 provider 实现 |
| `src/channels/` | 消息渠道实现 |
| `src/tools/` | tool 实现 |
| `src/memory/` | memory backend 与检索 |

## 开发约束（建议遵循）

1. 修改前先阅读上下文与现有测试。
2. 通过 vtable 扩展能力，避免跨层硬耦合。
3. 安全相关路径（`security/gateway/tools/runtime`）优先补边界测试。
4. 避免在无需求时引入新依赖或扩大配置面。
