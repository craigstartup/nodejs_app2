const express = require('express');
const { OpenAI } = require('openai');
const { Server } = require('socket.io');
const http = require('http');
const { Pinecone } = require('@pinecone-database/pinecone');

const app = express();
const server = http.createServer(app);
const io = new Server(server);
const port = process.env.PORT || 3000;

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

(async () => {
  // Initialize Pinecone
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });
  const pineconeIndex = pc.index('eoa');

  app.use(express.static('public'));

  async function generateEmbedding(text, model = 'text-embedding-ada-002') {
    try {
      const response = await openai.embeddings.create({
        input: text,
        model: model,
      });
      return response.data[0].embedding;
    } catch (error) {
      console.error('Error generating embedding:', error);
      return null;
    }
  }

  async function queryFromPinecone(vector, namespace, topK) {
    try {
      const queryResponse = await pineconeIndex.namespace(namespace).query({
        vector: vector,
        topK: topK,
        includeMetadata: true
      });
      return queryResponse;
    } catch (error) {
      console.error('Error querying Pinecone:', error);
      return null;
    }
  }

  io.on('connection', (socket) => {
    socket.on('sendPrompt', async ({ prompt, developerOutput, namespace, topK }) => {
      console.log("Received from client:", { prompt, developerOutput, namespace, topK });

      if (!prompt) {
        console.error('Invalid or missing data received from client:', { prompt });
        socket.emit('error', 'Invalid or missing prompt data');
        return;
      }

      try {
        // Generate embedding for the prompt
        const embedding = await generateEmbedding(prompt);

        // Query Pinecone for the closest matches
        const results = await queryFromPinecone(embedding, namespace, topK);

        if (!results || !results.matches) {
          console.error('No results returned from Pinecone');
          socket.emit('error', 'No results returned from Pinecone');
          return;
        }

        // Print the data received from Pinecone for troubleshooting
        console.log("Data received from Pinecone:", JSON.stringify(results, null, 2));

        // Extract all metadata fields dynamically
        const allMetadataFields = new Set();
        results.matches.forEach(match => {
          if (match.metadata) {
            Object.keys(match.metadata).forEach(field => {
              allMetadataFields.add(field);
            });
          }
        });
        const metadataFields = Array.from(allMetadataFields);
        console.log("Metadata fields:", metadataFields);

        // Construct context message
        const contextMessage = `The following metadata fields are included: ${metadataFields.join(', ')}.`;

        // Concatenate all transcripts and metadata into a single message
        const combinedMessage = results.matches.map(match => {
          const metadata = metadataFields.map(field => `${field}: ${match.metadata[field]}`).join("\n");
          return `${metadata}\nTranscript:\n${match.metadata.Transcript}\n\n`;
        }).join('');

        // Construct messages with context
        const messages = [
          { role: 'system', content: contextMessage },
          { role: 'user', content: combinedMessage },
          { role: 'user', content: prompt },
        ];

        if (developerOutput) {
          console.log("Messages sent to OpenAI:", JSON.stringify(messages, null, 2));
        }

        const completion = await openai.chat.completions.create({
          model: "gpt-4-turbo",
          messages: messages,
          stream: true,
        });

        for await (const chunk of completion) {
          socket.emit('responseChunk', {
            content: chunk.choices[0].delta.content,
            finish_reason: chunk.choices[0].finish_reason,
          });
          if (chunk.choices[0].finish_reason) {
            break;
          }
        }
      } catch (error) {
        console.error('Error processing prompt:', error);
        socket.emit('error', error.toString());
      }
    });
  });

  server.listen(port, () => {
    console.log(`Server is listening on port ${port}`);
  });
})();
