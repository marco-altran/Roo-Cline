import { Anthropic } from "@anthropic-ai/sdk";
import { Stream as AnthropicStream } from "@anthropic-ai/sdk/streaming";
import {
  anthropicDefaultModelId,
  AnthropicModelId,
  anthropicModels,
  ApiHandlerOptions,
  ModelInfo,
} from "../../shared/api";
import { ApiHandler } from "../index";
import { ApiStream } from "../transform/stream";
import { initializeLangSmith, traceableMethod } from "../langsmith/config";

// Define interface for usage tracking
interface Usage {
  inputTokens: number;
  outputTokens: number;
  cacheWriteTokens?: number | null;
  cacheReadTokens?: number | null;
}

// Initialize usage object
const createUsage = (): Usage => ({
  inputTokens: 0,
  outputTokens: 0
});

export class AnthropicHandler implements ApiHandler {
  private options: ApiHandlerOptions;
  private client: Anthropic;

  constructor(options: ApiHandlerOptions) {
    this.options = options;
    this.client = new Anthropic({
      apiKey: this.options.apiKey,
      baseURL: this.options.anthropicBaseUrl || undefined,
    });
    initializeLangSmith();
    
    // Bind the method to preserve 'this' context
    this.createMessage = this.createMessage.bind(this);
  }

  getModel(): { id: AnthropicModelId; info: ModelInfo } {
    const modelId = this.options.apiModelId;
    if (modelId && modelId in anthropicModels) {
      const id = modelId as AnthropicModelId;
      return { id, info: anthropicModels[id] };
    }
    return {
      id: anthropicDefaultModelId,
      info: anthropicModels[anthropicDefaultModelId],
    };
  }

  async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
    const tracedMethod = traceableMethod(
      async function*(
        this: AnthropicHandler,
        systemPrompt: string,
        messages: Anthropic.Messages.MessageParam[]
      ): ApiStream {
        let stream: AnthropicStream<Anthropic.Beta.PromptCaching.Messages.RawPromptCachingBetaMessageStreamEvent>;
        const modelId = this.getModel().id;
        let usage = createUsage();

        switch (modelId) {
          case "claude-3-5-sonnet-20241022":
          case "claude-3-5-haiku-20241022":
          case "claude-3-opus-20240229":
          case "claude-3-haiku-20240307": {
            const userMsgIndices = messages.reduce(
              (acc, msg, index) => (msg.role === "user" ? [...acc, index] : acc),
              [] as number[]
            );
            const lastUserMsgIndex = userMsgIndices[userMsgIndices.length - 1] ?? -1;
            const secondLastMsgUserIndex = userMsgIndices[userMsgIndices.length - 2] ?? -1;
            stream = await this.client.beta.promptCaching.messages.create(
              {
                model: modelId,
                max_tokens: this.getModel().info.maxTokens || 8192,
                temperature: 0,
                system: systemPrompt,
                messages: messages.map((message, index) => {
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
                                : content
                            ),
                    };
                  }
                  return message;
                }),
                stream: true,
              },
              {
                headers: { "anthropic-beta": "prompt-caching-2024-07-31" },
              }
            );
            break;
          }
          default: {
            stream = (await this.client.messages.create({
              model: modelId,
              max_tokens: this.getModel().info.maxTokens || 8192,
              temperature: 0,
              system: systemPrompt,
              messages,
              stream: true,
            })) as any;
            break;
          }
        }

        let accumulatedText = "";
        let currentBlockText = "";
        let isFirstBlock = true;

        for await (const chunk of stream) {
          switch (chunk.type) {
            case "message_start":
              usage = {
                inputTokens: chunk.message.usage?.input_tokens || 0,
                outputTokens: chunk.message.usage?.output_tokens || 0,
                cacheWriteTokens: chunk.message.usage?.cache_creation_input_tokens || undefined,
                cacheReadTokens: chunk.message.usage?.cache_read_input_tokens || undefined,
              };
              break;
            case "message_delta":
              if (chunk.usage?.output_tokens) {
                usage.outputTokens += chunk.usage.output_tokens;
              }
              break;
            case "message_stop":
              // If there's any remaining text, yield it
              if (accumulatedText) {
                yield {
                  type: "text",
                  text: accumulatedText,
                };
              }
              break;
            case "content_block_start":
              switch (chunk.content_block.type) {
                case "text":
                  if (!isFirstBlock) {
                    accumulatedText += "\n";
                  }
                  currentBlockText = chunk.content_block.text || "";
                  isFirstBlock = false;
                  break;
              }
              break;
            case "content_block_delta":
              switch (chunk.delta.type) {
                case "text_delta":
                  currentBlockText += chunk.delta.text;
                  break;
              }
              break;
            case "content_block_stop":
              accumulatedText += currentBlockText;
              currentBlockText = "";
              break;
          }
        }

        yield {
          type: "usage",
          inputTokens: usage.inputTokens,
          outputTokens: usage.outputTokens,
          cacheWriteTokens: usage.cacheWriteTokens || undefined,
          cacheReadTokens: usage.cacheReadTokens || undefined,
        };
      }.bind(this),
      { run_type: "llm", name: "Anthropic Chat" }
    );

    yield* tracedMethod(systemPrompt, messages);
  }
}
