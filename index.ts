import { Client, Events, GatewayIntentBits } from "discord.js";
import OpenAI from "openai";
import type { ResponsesModel } from "openai/resources/shared";
import { zodTextFormat } from "openai/helpers/zod";
import z from "zod";
import { guard } from "@libts/trycatch";
import { Logger } from "tslog";

const logger = new Logger();

const modelList = [
  "gpt-5-nano",
  "gpt-5-mini",
  "gpt-5.2",
  "gpt-5.1-codex",
] as const satisfies readonly ResponsesModel[];
const modelSchema = z.object({
  model: z.enum(modelList),
});

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});
const openai = new OpenAI();

client.once(Events.ClientReady, (c) => {
  logger.info(`Ready! Logged in as ${c.user.tag}`);
});

client.on(Events.MessageCreate, async (message) => {
  if (message.author.bot) return;
  if (!client.user) return;

  const prefix = `<@${client.user.id}>`;

  if (message.content.startsWith(prefix)) {
    const prompt = message.content.slice(prefix.length);
    if (prompt.length === 0) return;

    await message.channel.sendTyping();

    // Select the best model for the conversation
    const [selectResponse, selectError] = await guard(() =>
      openai.responses.parse({
        model: "gpt-5-nano",
        input: [
          {
            role: "system",
            content: [
              `Select the best model for the following conversation from ${modelList.join(", ")}.`,
              `Provide only the model name as response.`,
              `Select a smaller model for simpler conversations and a larger model for more complex ones.`,
            ].join("\n"),
          },
          { role: "user", content: prompt },
        ],
        text: {
          format: zodTextFormat(modelSchema, "model_schema"),
        },
      }),
    );
    if (selectError || !selectResponse) {
      console.error("Error selecting model:", selectError);
      return await message.reply(selectError.message);
    }

    // Generate the response using the selected model
    let history = [
      message,
      ...(
        await message.channel.messages.fetch({ limit: 5, before: message.id })
      ).values(),
    ].sort((a, b) => a.createdTimestamp - b.createdTimestamp);
    let input = history.map((m) => ({
      role: m.author.id === client.user!.id ? "assistant" : "user",
      content: [
        `from: ${m.author.displayName}`,
        `time: ${m.createdAt.toISOString()}`,
        m.content.replace(prefix, ""),
      ].join("\n"),
    })) satisfies OpenAI.Chat.Completions.ChatCompletionMessageParam[];

    for (let attempt = 0; attempt <= 2; attempt++) {
      await message.channel.sendTyping();
      const [response, error] = await guard(() =>
        openai.responses.create({
          model: selectResponse.output_parsed?.model ?? modelList[0],
          input: [
            {
              role: "system",
              content: [
                "You are a Discord assistant. You will be given only the last few messages.",
                "If the context is insufficient, use the tool `get_messages` to fetch more history before answering.",
                "Fetch the minimum amount needed.",
                "",
                `guild_name: ${message.guild ? message.guild.name : "Direct Message"}`,
                `channel_name: ${!message.channel.isDMBased() ? message.channel.name : "Direct Message"}`,
              ].join("\n"),
            },
            ...input,
          ],
          ...(attempt < 2
            ? {
                tools: [
                  {
                    type: "function",
                    name: "get_messages",
                    description:
                      "Fetch more message history from the current channel.",
                    parameters: {
                      type: "object",
                      properties: {
                        limit: {
                          type: "integer",
                          description:
                            "The number of messages to fetch, between 1 and 20.",
                          minimum: 1,
                          maximum: 20,
                        },
                      },
                      additionalProperties: false,
                      required: ["limit"],
                    },
                    strict: true,
                  },
                ],
              }
            : {}),
        }),
      );

      if (error || !response) {
        logger.error("Error generating response:", error);
        return await message.reply(error.message);
      }

      // If no tool was used, send the final response
      let didUseTool = response.output.some(
        (item) => item.type === "function_call" && item.name === "get_messages",
      );
      if (!didUseTool) {
        await message.reply({ content: response.output_text });
        logger.info({
          model: selectResponse.output_parsed?.model,
          history: history.map((h) => ({
            user: h.author.displayName,
            content: h.content,
          })),
          response: response.output_text,
        });
        return;
      }

      // Process tool calls
      await Promise.all(
        response.output.map(async (item) => {
          if (item.type === "function_call" && item.name === "get_messages") {
            history = [
              ...(
                await message.channel.messages.fetch({
                  limit: JSON.parse(item.arguments).limit,
                  before: history[history.length - 1]?.id,
                })
              ).values(),
              ...history,
            ].sort((a, b) => a.createdTimestamp - b.createdTimestamp);
            input = history.map((m) => ({
              role: m.author.id === client.user!.id ? "assistant" : "user",
              content: [
                `from: ${m.author.displayName}`,
                `time: ${m.createdAt.toISOString()}`,
                m.content.replace(prefix, ""),
              ].join("\n"),
            })) satisfies OpenAI.Chat.Completions.ChatCompletionMessageParam[];
          }
        }),
      );
    }
  }
});

client.login();
