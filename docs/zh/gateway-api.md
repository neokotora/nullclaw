# Gateway API

默认网关地址：`http://127.0.0.1:3000`

## 端点总览

| Endpoint | Method | 鉴权 | 说明 |
|---|---|---|---|
| `/health` | GET | 无 | 健康检查 |
| `/pair` | POST | `X-Pairing-Code` | 用一次性配对码换取 bearer token |
| `/webhook` | POST | `Authorization: Bearer <token>` | 发送消息：`{"message":"..."}` |
| `/whatsapp` | GET | Query 参数 | Meta Webhook 验证 |
| `/whatsapp` | POST | Meta 签名 | WhatsApp 入站消息 |

## 快速示例

### 1) 健康检查

```bash
curl http://127.0.0.1:3000/health
```

### 2) 配对换 token

```bash
curl -X POST \
  -H "X-Pairing-Code: 123456" \
  http://127.0.0.1:3000/pair
```

预期返回 bearer token（结构可能随版本调整）。

### 3) 发送 webhook 消息

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"hello from webhook"}' \
  http://127.0.0.1:3000/webhook
```

## 鉴权与安全建议

1. 保持 `gateway.require_pairing = true`。
2. 网关优先绑定 `127.0.0.1`，外网访问通过 tunnel/反向代理。
3. token 视为密钥，不写入公开仓库或日志。
