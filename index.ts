import { Client, Events, GatewayIntentBits, Message } from "discord.js";
import OpenAI from "openai";
import type { ResponsesModel } from "openai/resources/shared";
import { zodTextFormat } from "openai/helpers/zod";
import z from "zod";
import { guard } from "@libts/trycatch";
import { Logger } from "tslog";

const logger = new Logger();

process.on("uncaughtException", logger.error);
process.on("unhandledRejection", logger.error);

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

function buildTextFromDiscordMessage(m: Message, prefix: string) {
  return [
    `from: ${m.author.displayName}`,
    `time: ${m.createdAt.toISOString()}`,
    (m.content ?? "").startsWith(prefix)
      ? (m.content ?? "").slice(prefix.length).trimStart()
      : (m.content ?? ""),
  ].join("\n");
}

async function buildPartsFromDiscordMessage(m: Message, prefix: string) {
  const parts: OpenAI.Responses.ResponseInputContent[] = [];

  parts.push({
    type: "input_text",
    text: buildTextFromDiscordMessage(m, prefix),
  });

  (
    await Promise.all(
      [...m.attachments.values()]
        .slice(0, 10)
        .map(async (f): Promise<OpenAI.Responses.ResponseInputContent> => {
          return f.contentType?.startsWith("image/")
            ? {
                type: "input_image",
                image_url: f.url,
                detail: "auto",
              }
            : f.contentType?.startsWith("text/")
              ? {
                  type: "input_text",
                  text: [
                    `from: ${m.author.displayName}`,
                    `time: ${m.createdAt.toISOString()}`,
                    `filename: ${f.name}`,
                    `content:`,
                    await fetch(f.url).then((res) => res.text()),
                  ].join("\n"),
                }
              : {
                  type: "input_file",
                  file_url: f.url,
                };
        }),
    )
  ).forEach((part) => parts.push(part));

  return parts;
}

client.once(Events.ClientReady, (c) => {
  logger.info(`Ready! Logged in as ${c.user.tag}`);
});

client.on(Events.MessageCreate, async (message) => {
  if (message.author.bot) return;
  if (!client.user) return;

  const prefix = `<@${client.user.id}>`;
  if (
    message.content.startsWith(prefix) ||
    (message.reference !== null &&
      (await message.fetchReference())?.author.id === client.user.id)
  ) {
    const typingInterval = setInterval(
      () => message.channel.sendTyping(),
      3000,
    );
    try {
      const prompt = message.content.slice(
        message.content.startsWith(prefix) ? prefix.length : 0,
      );
      if (prompt.length === 0) return;

      // Select the best model for the conversation
      logger.debug("Selecting model...");
      const [selectResponse, selectError] = await guard(async () =>
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
            {
              role: "user",
              content: await buildPartsFromDiscordMessage(message, prefix),
            },
          ],
          text: {
            format: zodTextFormat(modelSchema, "model_schema"),
          },
        }),
      );
      if (selectError || !selectResponse) {
        logger.error("Error selecting model:", selectError);
        return await message.reply(selectError.message);
      }
      logger.info("Selected model:", selectResponse.output_parsed?.model);

      // Generate the response using the selected model
      let history = [
        message,
        ...(
          await message.channel.messages.fetch({
            limit: 5,
            before: message.reference
              ? message.reference.messageId
              : message.id,
          })
        ).values(),
      ].sort((a, b) => a.createdTimestamp - b.createdTimestamp);

      for (let attempt = 0; attempt <= 2; attempt++) {
        // Build the input
        logger.debug("Building input...");
        let input = (await Promise.all(
          history.map(async (m) => ({
            role: m.author.id === client.user!.id ? "assistant" : "user",
            content:
              m.author.id === client.user!.id
                ? await buildTextFromDiscordMessage(m, prefix)
                : await buildPartsFromDiscordMessage(m, prefix),
          })),
        )) satisfies OpenAI.Responses.ResponseInput;

        logger.debug("Generating response...");
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

        if (
          !response.output.some(
            (item) =>
              item.type === "function_call" && item.name === "get_messages",
          )
        ) {
          for (let i = 0; i < response.output_text.length; i += 2000) {
            if (i === 0)
              await message.reply({
                content: response.output_text.slice(i, i + 2000),
              });
            else
              await message.channel.send({
                content: response.output_text.slice(i, i + 2000),
              });
          }

          logger.info({
            model: selectResponse.output_parsed?.model,
            input,
            response: response.output_text,
          });
          return;
        }

        logger.debug("Processing tool calls: Picking up more history");

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
            }
          }),
        );
      }
    } catch (err) {
      logger.error("Unexpected error:", err);
      await message.reply(
        err instanceof Error ? err.message : "An unexpected error occurred.",
      );
    } finally {
      logger.debug("End Processing message");
      clearInterval(typingInterval);
    }
  }
});

client.login();
