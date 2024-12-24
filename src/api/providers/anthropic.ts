import { Anthropic } from "@anthropic-ai/sdk"
import { Stream as AnthropicStream } from "@anthropic-ai/sdk/streaming"
import {
  anthropicDefaultModelId,
  AnthropicModelId,
  anthropicModels,
  ApiHandlerOptions,
  ModelInfo,
} from "../../shared/api"
import { ApiHandler } from "../index"
import { ApiStream } from "../transform/stream"
import { initializeLangSmith, wrapAnthropicClient, traceableMethod } from "../langsmith/config"

export class AnthropicHandler implements ApiHandler {
  private options: ApiHandlerOptions
  private client: Anthropic

  constructor(options: ApiHandlerOptions) {
    this.options = options
    initializeLangSmith()
    this.client = wrapAnthropicClient(new Anthropic({
      apiKey: this.options.apiKey,
      baseURL: this.options.anthropicBaseUrl || undefined,
    }))

    // Bind the method to preserve 'this' context
    this.createMessage = this.createMessage.bind(this)
  }

  async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
    const tracedMethod = traceableMethod(async function*(
      this: AnthropicHandler,
      inputs: { systemPrompt: string; messages: Anthropic.Messages.MessageParam[] }
    ): ApiStream {
      let stream: AnthropicStream<Anthropic.Beta.PromptCaching.Messages.RawPromptCachingBetaMessageStreamEvent>
      const modelId = this.getModel().id
      let fullText = ""
      let usageData = {
        inputTokens: 0,
        outputTokens: 0,
        cacheWriteTokens: undefined as number | undefined,
        cacheReadTokens: undefined as number | undefined,
      }

      switch (modelId) {
        // 'latest' alias does not support cache_control
        case "claude-3-5-sonnet-20241022":
        case "claude-3-5-haiku-20241022":
        case "claude-3-opus-20240229":
        case "claude-3-haiku-20240307": {
          const userMsgIndices = inputs.messages.reduce(
            (acc, msg, index) => (msg.role === "user" ? [...acc, index] : acc),
            [] as number[],
          )
          const lastUserMsgIndex = userMsgIndices[userMsgIndices.length - 1] ?? -1
          const secondLastMsgUserIndex = userMsgIndices[userMsgIndices.length - 2] ?? -1
          stream = await this.client.beta.promptCaching.messages.create(
            {
              model: modelId,
              max_tokens: this.getModel().info.maxTokens || 8192,
              temperature: 0,
              system: [{ text: inputs.systemPrompt, type: "text", cache_control: { type: "ephemeral" } }],
              messages: inputs.messages.map((message, index) => {
                if (index === lastUserMsgIndex || index === secondLastMsgUserIndex) {
                  return {
                    ...message,
                    content:
                      typeof message.content === "string"
                        ? [
                            {
                              type: "text",
                              text: message.content,
                              cache_control: { type: "ephemeral" },
                            },
                          ]
                        : message.content.map((content, contentIndex) =>
                            contentIndex === message.content.length - 1
                              ? { ...content, cache_control: { type: "ephemeral" } }
                              : content,
                          ),
                  }
                }
                return message
              }),
              stream: true,
            },
            (() => {
              switch (modelId) {
                case "claude-3-5-sonnet-20241022":
                case "claude-3-5-haiku-20241022":
                case "claude-3-opus-20240229":
                case "claude-3-haiku-20240307":
                  return {
                    headers: { "anthropic-beta": "prompt-caching-2024-07-31" },
                  }
                default:
                  return undefined
              }
            })(),
          )
          break
        }
        default: {
          stream = (await this.client.messages.create({
            model: modelId,
            max_tokens: this.getModel().info.maxTokens || 8192,
            temperature: 0,
            system: [{ text: inputs.systemPrompt, type: "text" }],
            messages: inputs.messages,
            stream: true,
          })) as any
          break
        }
      }

      for await (const chunk of stream) {
        switch (chunk.type) {
          case "message_start":
            const msgUsage = chunk.message.usage
            usageData = {
              inputTokens: msgUsage.input_tokens || 0,
              outputTokens: msgUsage.output_tokens || 0,
              cacheWriteTokens: msgUsage.cache_creation_input_tokens || undefined,
              cacheReadTokens: msgUsage.cache_read_input_tokens || undefined,
            }
            yield {
              type: "usage",
              ...usageData
            }
            break
          case "message_delta":
            usageData.outputTokens = chunk.usage.output_tokens || 0
            break
          case "message_stop":
            break
          case "content_block_start":
            switch (chunk.content_block.type) {
              case "text":
                if (chunk.index > 0) {
                  fullText += "\n"
                  yield {
                    type: "text",
                    text: "\n",
                  }
                }
                fullText += chunk.content_block.text
                yield {
                  type: "text",
                  text: chunk.content_block.text,
                }
                break
            }
            break
          case "content_block_delta":
            switch (chunk.delta.type) {
              case "text_delta":
                fullText += chunk.delta.text
                yield {
                  type: "text",
                  text: chunk.delta.text,
                }
                break
            }
            break
          case "content_block_stop":
            break
        }
      }

      // Return the final completion for LangSmith tracing
      return {
        inputs: {
          systemPrompt: inputs.systemPrompt,
          messages: inputs.messages,
          model: modelId,
        },
        outputs: {
          content: fullText,
          usage: usageData,
        }
      }
    }.bind(this), { run_type: "llm", name: "Anthropic Chat" })

    yield* tracedMethod({ systemPrompt, messages })
  }

  getModel(): { id: AnthropicModelId; info: ModelInfo } {
    const modelId = this.options.apiModelId
    if (modelId && modelId in anthropicModels) {
      const id = modelId as AnthropicModelId
      return { id, info: anthropicModels[id] }
    }
    return { id: anthropicDefaultModelId, info: anthropicModels[anthropicDefaultModelId] }
  }
}
