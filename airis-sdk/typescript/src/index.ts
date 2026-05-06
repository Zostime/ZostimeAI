/*
参考了 Neuro API
https://github.com/VedalAI/neuro-sdk/blob/main/API/SPECIFICATION.md
*/

import { WebSocket } from 'ws';

/**
 * Airis API 中的可注册动作。
 */
export interface Action {
    /** 动作的唯一标识符. 小写字母, 单词间用下划线或短横线分隔. */
    name: string;
    /** 动作的自然语言描述, 会直接呈现给 Airis. */
    description: string;
    /**
     * 可选的 JSON Schema 对象，描述动作参数的结构。
     * 必须为 {"type": "object", ...} 格式。
     */
    schema?: {
        type: "object";
    } & Record<string, unknown>;
}

export interface ActionCommand {
    id: string;
    name: string;
    data: string;
}

// noinspection JSUnusedGlobalSymbols
export class Websocket {

    uri: string | null = null;
    ws: WebSocket | null = null;
    gameName: string | null = null;
    private _actionCallback: ((action: ActionCommand) => void | Promise<void>) | null = null;
    private _pendingActions = new Map<
        string,
        {
            resolve: (action: ActionCommand) => void;
            reject: (reason?: any) => void;
        }
    >();

    /**
     * 这条消息应在游戏开始时立即发送, 以告知 Airis 游戏正在运行.
     *
     * 这个消息会清除游戏之前注册的所有操作并进行初始设置, 因此应该是你发送的第一条消息.
     * @param gameName - 游戏名称. 这用来识别游戏. 它应该始终保持不变, 不应改变.
     *                   你应该使用游戏的显示名称, 包括任何空格和符号(例如 "Buckshot Roulette").
     *                   服务器不会包含这个字段.
     */
    async startup(gameName: string): Promise<void> {
        if (!this.ws) {
            throw new Error("WebSocket not connected. Call connect() first.");
        }
        this.gameName = gameName;
        const payload = {
            command: "startup",
            game: gameName
        };

        this.ws.send(JSON.stringify(payload));
    }

    /**
     * 这条消息可以用来通知 Airis 游戏中正在发生的事情。
     *
     * @param message - 一条描述游戏中发生情况的纯文本信息.这些信息将直接由 Airis 接收.
     * @param silent - 若为 True, 消息将被添加到 Airis 的上下文中, 而不会提示她对此作出回应.
     *                 若为 False, Airis 可能会直接回应消息, 除非她正忙于与其他人交谈或聊天.
     */
    async sendContext(message: string, silent: boolean = false): Promise<void> {
        if (!this.ws) {
            throw new Error("WebSocket not connected. Call connect() first.");
        }
        const payload = {
            command: "context",
            game: this.gameName,
            data: {
                message: message,
                silent: silent
            }
        };

        this.ws.send(JSON.stringify(payload));
    }

    /**
     * 此消息为 Airis 注册一个或多个操作以供使用.
     *
     * @param actions - 一组需要注册的动作. 如果你尝试注册已经注册的动作, 它会被忽略.
     */
    async registerActions(actions: Action[]): Promise<void> {
        if (!this.ws) {
            throw new Error("WebSocket not connected. Call connect() first.");
        }

        const payload = {
            command: 'actions/register',
            game: this.gameName,
            data: {
                actions: actions.map(a => ({
                    name: a.name,
                    description: a.description,
                    ...(a.schema !== undefined ? {schema: a.schema} : {})
                }))
            }
        };

        this.ws.send(JSON.stringify(payload));
    }

    /**
     * 该消息会取消注册一个或多个动作, 阻止 Airis 继续使用它们。
     *
     * @param actionNames - 要注销的操作名称. 如果你尝试注销一个未注册的操作, 不会有任何问题.
     */
    async unregisterActions(actionNames: string[]): Promise<void> {
        if (!this.ws) {
            throw new Error("WebSocket not connected. Call connect() first.");
        }

        const payload = {
            command: "actions/unregister",
            game: this.gameName,
            data: {
                action_names: actionNames
            }
        };

        this.ws.send(JSON.stringify(payload));
    }

    /**
     * 强制 Airis 立即从给定的动作列表中选择一个执行。
     * 服务器将构建一个临时的决策上下文, 要求 Airis 必须返回一个工具调用.
     *
     * @param query - 告诉 Airis 当前应该做什么的指令(例如 "It is your turn. Please place an O.").
     * @param actionNames - 限定 Airis 只能从中选择的动作名称列表。
     * @param state - 可选,描述当前游戏完整状态的字符串(支持 Markdown 或 JSON).
     * @param ephemeralContext - 若为 True, 此次强制请求的状态和指令在动作完成后会被遗忘;
     *                           若为 False, 信息会保留在 Airis 的长期上下文中.
     * @param priority - 决定 Airis 回应紧急程度的优先级。可选值:
     *                   - "low"：等待 Airis 说完当前的话再处理.
     *                   - "medium"：让 Airis 尽快结束当前话语.
     *                   - "high"：缩短 Airis 当前话语并立即处理.
     *                   - "critical"：立即打断 Airis 说话并处理 (谨慎使用).
     */
    async forceActions(
        query: string,
        actionNames: string[],
        state: string = "",
        ephemeralContext: boolean = false,
        priority: "low" | "medium" | "high" | "critical" = "low"
    ): Promise<void> {
        if (!this.ws) {
            throw new Error("WebSocket not connected. Call connect() first.");
        }
        const payload = {
            command: "actions/force",
            game: this.gameName,
            data: {
                state: state,
                query: query,
                ephemeral_context: ephemeralContext,
                priority: priority,
                action_names: actionNames
            }
        };

        this.ws.send(JSON.stringify(payload));
    }

    /**
     * 将 Airis 要求执行的动作结果返回给服务器.
     *
     * @param actionId - 从 Airis 下发的 action 命令中获取的唯一 ID。
     * @param success - 动作是否执行成功。若为 False 且该动作属于一次 force 请求，
     *                  整个 force 流程会被立即重试。
     * @param message - 可选的附加信息。成功时可提供简短上下文提示；失败时应包含错误原因。
     */
    async sendActionResult(actionId: string, success: boolean, message: string = ""): Promise<void> {
        if (!this.ws) {
            throw new Error("WebSocket not connected. Call connect() first.");
        }

        const payload = {
            "command": "action/result",
            "game": this.gameName,
            "data": {
                "id": actionId,
                "success": success,
                "message": message
            }
        };

        this.ws.send(JSON.stringify(payload));
    }

    /**
     * 注册一个回调函数，用于处理服务器下发的 action 命令。
     * 回调函数应接收一个包含以下字段的对象，包含：
     * - id: string 本次动作调用的唯一标识
     * - name: string AI 决定执行的动作名称
     * - data: string 包含动作参数的 JSON 字符串
     *
     * @param callback - 处理 action 的回调，可以是同步或异步函数
     */
    onAction(callback: ((action: ActionCommand) => void | Promise<void>)): void {
        this._actionCallback = callback;
    }

    /**
     * 内部监听，持续接收 WebSocket 消息并处理。
     * 在 connect() 成功之后调用，不应手动调用。
     */
    private _listen(): void {
        if (!this.ws) return;

        this.ws.on('message', async (data: Buffer) => {
            try {
                const msg = JSON.parse(data.toString());
                if (msg.command === 'action') {
                    const payload = msg.data;
                    const id = payload.id;

                    const pending = this._pendingActions.get(id);
                    if (pending) {
                        pending.resolve(payload);
                        this._pendingActions.delete(id);
                        return;
                    }

                    if (this._actionCallback) {
                        await this._actionCallback(payload);
                    }
                }
            } catch {
            }
        });

        this.ws.on('close', () => {
            for (const [, pending] of this._pendingActions) {
                pending.reject(new Error("WebSocket disconnected"));
            }
            this._pendingActions.clear();
            this.ws = null;
        });

        this.ws.on('error', () => {});
    }

    async connect(uri: string): Promise<void> {
        this.uri = uri;
        const ws = new WebSocket(uri);
        this.ws = ws;
        await new Promise<void>((resolve, reject) => {
            ws.onopen = () => resolve();
            ws.onerror = (err: any) => {
                reject(new Error(`连接失败: ${err?.message || err}`));
            };
        });
        this._listen();
    }

    disconnect(): void {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        for (const [, p] of this._pendingActions) {
            p.reject(new Error("WebSocket disconnected"));
        }
        this._pendingActions.clear();
    }

    isConnected(): boolean {
        return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
    }
}