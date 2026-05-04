# Airis API 规范

## 归属说明

本规范基于 Neuro API 规范进行修改和扩展，以适配 Airis。

- 原始项目：Neuro API
- 源代码：https://github.com/VedalAI/neuro-sdk
- 许可证：MIT 许可证

本项目与原 Neuro API 项目无直接关联。

游戏（客户端）发送给 Airis（服务端）的消息格式如下：

```
{
    "command": string,
    "game": string,
    "data": { 
        [key: string]: any
    }?
}
```

Airis（服务端）发送给游戏（客户端）的消息格式如下：

```
{
    "command": string,
    "data": { 
        [key: string]: any
    }?
}
```

> [!Warning]
> WebSocket 消息必须使用纯文本格式（而非二进制）进行发送和接收。

#### 参数

- `command`：WebSocket 命令。请参阅下面列出的命令列表。
- `game`：游戏名称，用于标识游戏。它应始终相同且不应更改。请使用游戏的显示名称，包括空格和符号（例如 `"Buckshot Roulette"`）。服务端不会包含此字段。
- `data`：命令数据。该对象会根据发送/接收的命令而有所不同，有些命令可能没有任何数据，此时该字段可能为 `undefined` 或空对象 `{}`，具体取决于实现。

## 常见类型

以下数据类型在整个 API 中使用。

### 动作

动作是一种可注册的命令，Airis 可以随时执行。

```
{
    "name": string,
    "description": string,
    "schema": {
        "type": "object",
        [key: string]: any
    }?
}
```

#### 参数

- `name`：动作的名称，也是其唯一标识符。应使用小写字符串，单词之间用下划线或短横线分隔（例如 `"join_friend_lobby"`、`"use_item"`）。
- `description`：对该动作功能的纯文本描述。**该信息将直接提供给 Airis。**
- `schema`：一个有效的简单 JSON Schema 对象，用于描述响应数据的结构。如果您的动作没有参数，可以省略此字段或将其设为 `{}`。如果提供 Schema，则其 `type` 必须为 `"object"`。**该信息将直接提供给 Airis。**

## 客户端 → 服务端（C2S，游戏到 Airis）

### 启动

此消息应在游戏启动后立即发送，以便让 Airis 知道游戏正在运行。此消息将清除该游戏之前注册的所有动作并进行初始设置，因此应作为您发送的第一条消息。

```
{
    "command": "startup",
    "game": string
}
```

### 上下文

可以发送此消息，用于向 Airis 提供游戏中的上下文信息

```
{
    "command": "context",
    "game": string,
    "data": {
        "message": string,
        "silent": boolean
    }
}
```

#### 参数

- `message`：描述游戏中正在发生的事情的纯文本消息。**该信息将直接提供给 Airis。**
- `silent`：如果为 `true`，该消息将添加到 Airis 的上下文中，但不会触发响应。如果为 `false`，Airis 可能会对此消息进行响应，但不保证一定响应。，除非她正忙于与他人或聊天对话。

### 注册动作

此消息为 Airis 注册一个或多个动作以供使用。

```
{
    "command": "actions/register",
    "game": string,
    "data": {
        "actions": Action[]
    }
}
```

#### 参数

- `actions`：要注册的动作数组。如果尝试注册已注册的动作，则会被忽略。

### 注销动作

此消息注销一个或多个动作，阻止 Airis 继续使用它们。

```
{
    "command": "actions/unregister",
    "game": string,
    "data": {
        "action_names": string[]
    }
}
```

#### 参数

- `action_names`：要注销的动作名称。如果尝试注销未注册的动作，不会有任何问题。

### 强制执行动作

此消息强制 Airis 尽快执行所列动作之一。注意，如果她正在说话，这可能需要一点时间。

> [!Important]
> 重要提示：Airis 一次只能处理一个动作强制请求。
> 在另一个强制请求进行中时发送强制请求会导致问题！

```
{
    "command": "actions/force",
    "game": string,
    "data": {
        "state": string?,
        "query": string,
        "ephemeral_context": boolean?, // 默认为 false
        "priority": "low" | "medium" | "high" | "critical", // 默认为 "low"
        "action_names": string[]
    }
}
```

#### 参数

- `state`：描述游戏当前状态的任意字符串。可以是纯文本、JSON、Markdown 或其他任何格式。我们推荐使用 Markdown。**该信息将直接提供给 Airis。**
- `query`：一条纯文本消息，告诉 Airis 当前她应该做什么（例如 `"It is now your turn. Please perform an action. If you want to use any items, you should use them before picking up the shotgun."`）。**该信息将直接提供给 Airis。**
- `ephemeral_context`：如果为 `false`，在强制动作完成后，Airis 会记住 `state` 和 `query` 参数中提供的上下文。如果为 `true`，Airis 只会在强制动作期间记住该上下文。
- `priority`：确定 Airis 在说话时对动作强制请求的响应紧急程度。如果 Airis 不在说话，此设置无效。默认为 `"low"`，会使 Airis 等待直到说完话再响应。`"medium"` 会促使她更快结束当前语句。`"high"` 会提示她立即处理动作强制请求，缩短语句后响应。`"critical"` 会打断她的说话并立即响应。请谨慎使用 `"critical"`，因为它可能导致突然且可能令人不适的中断。
- `action_names`：Airis 应从中选择的动作名称。

### 动作结果

此消息需在动作验证后尽快发送，以便让 Airis 继续。

> [!Important]
> 重要提示：在您发送动作结果之前，Airis 会一直等待她的动作结果！
> 请务必尽快发送。通常应在验证动作参数之后、实际在游戏中执行动作之前发送。

```
{
    "command": "action/result",
    "game": string,
    "data": {
        "id": string,
        "success": boolean,
        "message": string?
    }
}
```

#### 参数

- `id`：此结果对应的动作 ID。直接从动作消息中获取。
- `success`：动作是否成功。*如果为 `false` 且该动作是强制动作的一部分，则整个强制动作将被 Airis 立即重试。*
- `message`：描述动作执行情况的纯文本消息。如果不成功，应是一条错误消息。如果成功，可以是空消息，也可以提供一小段上下文给 Airis，关于她刚刚执行的动作（例如 `"Remember to not share this with anyone."`）。**该信息将直接提供给 Airis。**

> [!Tip]
> 提示：由于将 `success` 设为 `false` 会导致重试强制动作（若存在），如果动作不成功但您不想重试，应将 `success` 设为 `true` 并在 `message` 字段中提供错误消息。

## 服务端 → 客户端（S2C，Airis 到游戏）

### 动作

当 Airis 尝试执行动作时，会发送此消息。您应尽快用动作结果进行响应。

```
{
    "command": "action",
    "data": {
        "id": string,
        "name": string,
        "data": string?
    }
}
```

#### 参数

- `id`：动作的唯一 ID。在发回动作结果时应使用它。
- `name`：Airis 尝试执行的动作名称。
- `data`：Airis 发送的动作数据的 JSON 字符串。它***应*** 是一个与注册动作时提供的 JSON Schema 匹配的对象。如果您未提供 Schema，此参数通常为 `undefined`。

> [!Caution]
> 注意：`data` 参数直接来自 Airis，因此有可能会格式错误、包含无效 JSON 或与提供的 Schema 不完全匹配。
> 您有责任验证 JSON，如果无效，应返回一个不成功的动作结果。