# 安全机制

NullClaw 默认走 secure-by-default：本地绑定、配对鉴权、沙箱隔离、最小权限。

## 基线能力

| 项 | 状态 | 说明 |
|---|---|---|
| 网关默认不公网暴露 | 已启用 | 默认绑定 `127.0.0.1`；无 tunnel/显式放开时拒绝公网绑定 |
| 配对鉴权 | 已启用 | 启动时一次性 6 位 pairing code，`POST /pair` 换 token |
| 文件系统范围限制 | 已启用 | 默认 `workspace_only = true`，阻止越界访问 |
| 隧道访问控制 | 已启用 | 公网场景优先通过 Tailscale/Cloudflare/ngrok/custom tunnel |
| 沙箱隔离 | 已启用 | 自动选择 Landlock/Firejail/Bubblewrap/Docker |
| 密钥加密 | 已启用 | 凭据采用 ChaCha20-Poly1305 本地加密存储 |
| 资源限制 | 已启用 | 可配置内存/CPU/子进程等限制 |
| 审计日志 | 已启用 | 可开启并设置保留策略 |

## Channel allowlist 规则

- `allow_from: []`：拒绝所有入站。
- `allow_from: ["*"]`：允许所有来源（高风险，仅显式确认后使用）。
- 其他：按精确匹配允许列表。

## Nostr 特殊规则

- `owner_pubkey` 始终允许（即使 `dm_allowed_pubkeys` 更严格）。
- 私钥使用 `enc2:` 加密格式落盘，仅运行时解密到内存；停止 channel 后清理。

## 推荐安全配置

```json
{
  "gateway": {
    "host": "127.0.0.1",
    "port": 3000,
    "require_pairing": true,
    "allow_public_bind": false
  },
  "autonomy": {
    "level": "supervised",
    "workspace_only": true,
    "max_actions_per_hour": 20
  },
  "security": {
    "sandbox": { "backend": "auto" },
    "audit": { "enabled": true, "retention_days": 90 }
  }
}
```

## 高风险配置提醒

以下配置会显著扩大权限边界，应仅用于受控环境：

- `autonomy.level = "full"`
- `allowed_commands = ["*"]`
- `allowed_paths = ["*"]`
- `gateway.allow_public_bind = true`
