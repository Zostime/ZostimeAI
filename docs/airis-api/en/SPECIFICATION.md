# Airis API Specification

## Attribution

This specification is derived from the Neuro API specification,
with modifications and extensions for the Airis.

- Original Project: Neuro API
- Source: https://github.com/VedalAI/neuro-sdk
- License: MIT License

This project is not affiliated with the original Neuro API project.

Messages sent by the game (client) to Airis (server) have the following format:
```
{
    "command": string,
    "game": string,
    "data": { 
        [key: string]: any
    }?
}
```

Messages sent by Airis (server) to the game (client) have the following format:
```
{
    "command": string,
    "data": { 
        [key: string]: any
    }?
}
```

> [!Warning]  
> Websocket messages are sent and received in plaintext format (not binary).  

#### Parameters
- `command`: The websocket command. Look below for a list of commands.
- `game`: The game name. This is used to identify the game. It should _always_ be the same and should not change. You should use the game's display name, including any spaces and symbols (e.g. `"Buckshot Roulette"`). The server will not include this field.
- `data`: The command data. This object is different depending on which command you are sending/receiving, and some commands may not have any data, in which case this object will be either `undefined` or `{}`.

## Common Types

The following data types are used throughout the API.

### Action

An action is a registerable command that Airis can execute whenever she wants.

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

#### Parameters
- `name`: The name of the action, which is its _unique identifier_. This should be a lowercase string, with words separated by underscores or dashes (e.g. `"join_friend_lobby"`, `"use_item"`).
- `description`: A plaintext description of what this action does. **This information will be directly received by Airis.**
- `schema`: A **valid** simple JSON schema object that describes how the response data should look like. If your action does not have any parameters, you can omit this field or set it to `{}`. If you provide a schema, it must be of `type` `"object"`. **This information will be directly received by Airis.**

## Outgoing Messages (C2S, Game to Airis)

### Startup

This message should be sent as soon as the game starts, to let Airis know that the game is running.

This message clears all previously registered actions for this game and does initial setup, and as such should be the very first message that you send.

```
{
    "command": "startup",
    "game": string
}
```

### Context

This message can be sent to let Airis know about something that is happening in game.

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

#### Parameters
- `message`: A plaintext message that describes what is happening in the game. **This information will be directly received by Airis.**
- `silent`: If `true`, the message will be added to Airis's context without prompting her to respond to it. If `false`, Airis _might_ respond to the message directly, unless she is busy talking to someone else or to chat.

### Register Actions

This message registers one or more actions for Airis to use.

```
{
    "command": "actions/register",
    "game": string,
    "data": {
        "actions": Action[]
    }
}
```

#### Parameters
- `actions`: An array of actions to be registered. If you try to register an action that is already registered, it will be ignored.

### Unregister Actions

This message unregisters one or more actions, preventing Airis from using them anymore.

```
{
    "command": "actions/unregister",
    "game": string,
    "data": {
        "action_names": string[]
    }
}
```

#### Parameters
- `action_names`: The names of the actions to unregister. If you try to unregister an action that isn't registered, there will be no problem.

### Force Actions

This message forces Airis to execute one of the listed actions as soon as possible. Note that this might take a bit if she is already talking.

> [!Important]  
> Airis can only handle one action force at a time.  
> Sending an action force while another one is in progress will cause problems!

```
{
    "command": "actions/force",
    "game": string,
    "data": {
        "state": string?,
        "query": string,
        "ephemeral_context": boolean?, // Defaults to false
        "priority": "low" | "medium" | "high" | "critical", // Defaults to "low"
        "action_names": string[]
    }
}
```

#### Parameters
- `state`: An arbitrary string that describes the current state of the game. This can be plaintext, JSON, Markdown, or any other format. We recommend using Markdown. **This information will be directly received by Airis.**
- `query`: A plaintext message that tells Airis what she is currently supposed to be doing (e.g. `"It is now your turn. Please perform an action. If you want to use any items, you should use them before picking up the shotgun."`). **This information will be directly received by Airis.**
- `ephemeral_context`: If `false`, the context provided in the `state` and `query` parameters will be remembered by Airis after the actions force is completed. If `true`, Airis will only remember it for the duration of the actions force.
- `priority`: Determines how urgently Airis should respond to the action force when she is speaking. If Airis is not speaking, this setting has no effect. The default is `"low"`, which will cause Airis to wait until she finishes speaking before responding. `"medium"` causes her to finish her current utterance sooner. `"high"` prompts her to process the action force immediately, shortening her utterance and then responding. `"critical"` will interrupt her speech and make her respond at once. Use `"critical"` with caution, as it may lead to abrupt and potentially jarring interruptions.
- `action_names`: The names of the actions that Airis should choose from.

### Action Result

This message needs to be sent as soon as possible after an action is validated, to allow Airis to continue.

> [!Important]  
> Until you send an action result, Airis will just be waiting for the result of her action!  
> Please make sure to send this as soon as possible.   
> It should usually be sent after validating the action parameters, before it is actually executed in-game

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

#### Parameters
- `id`: The id of the action that this result is for. This is grabbed from the action message directly.
- `success`: Whether the action was successful. _If this is `false` and this action is part of an actions force, the whole actions force will be immediately retried by Airis._
- `message`: A plaintext message that describes what happened when the action was executed. If not successful, this should be an error message. If successful, this can either be empty, or provide a _small_ context to Airis regarding the action she just took (e.g. `"Remember to not share this with anyone."`). **This information will be directly received by Airis.**

> [!Tip]  
> Since setting `success` to `false` will retry the action force if there was one, if the action was not successful, but you don't want it to be retried, you should set `success` to `true` and provide an error message in the `message` field.

## Incoming Messages (S2C, Airis to Game)

### Action

This message is sent by Airis when she tries to execute an action. You should respond to it with an action result as soon as possible.

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

#### Parameters
- `id`: A unique id for the action. You should use it when sending back the action result.
- `name`: The name of the action that Airis is trying to execute.
- `data`: The JSON-stringified data for the action, as sent by Airis. This **_should\*_** be an object that matches the JSON schema you provided when registering the action. If you did not provide a schema, this parameter will usually be `undefined`.

> [!Caution]  
> The `data` parameter comes directly from Airis, so there is a chance it might be malformed, contain invalid JSON, or not match the provided schema exactly.  
> You are responsible for validating the JSON and returning an unsuccessful action result if it is invalid.
