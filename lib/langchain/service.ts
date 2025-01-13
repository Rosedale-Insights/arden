import { Pinecone } from "@pinecone-database/pinecone";
import { OpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PineconeStore } from "@langchain/pinecone";
import { Document } from "@langchain/core/documents";
import { MultiQueryRetriever } from "langchain/retrievers/multi_query";
import { VectorStoreRetriever } from "@langchain/core/vectorstores";
import { PromptTemplate } from "@langchain/core/prompts";

export class LangChainService {
  private pineconeClient: Pinecone;
  private embeddings: OpenAIEmbeddings;
  private vectorStore!: PineconeStore;
  private retriever!: VectorStoreRetriever;
  private userId!: string;
  private llm = new OpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });
  private queryGenerationPrompt = PromptTemplate.fromTemplate(
    `Rosedale is a company that provides a 360 coaching service for executives and founders. Our customers that go through this service are looking for leadership advice and guidance.

    Original question: {question}

    Alternative questions:
    Consider:
    - Direct quotes from 360 feedback and self-reflections as evidence
    - Specific examples with supporting quotes
    - Verbatim feedback that supports key points
    - Exact phrases and statements that demonstrate impact
    - Clear examples backed by participant quotes
    - Patterns in feedback with supporting evidence

    For each insight, find relevant quotes. Clean up any grammar or formatting issues in the quotes while preserving their meaning.`
  );

  constructor() {
    this.pineconeClient = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });
    this.embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
      modelName: "text-embedding-3-large",
      dimensions: 3072,
    });
  }

  async initialize(userId: string) {
    this.userId = userId;
    if (this.vectorStore) return;

    const index = this.pineconeClient.Index(process.env.PINECONE_INDEX_NAME!);

    this.vectorStore = await PineconeStore.fromExistingIndex(this.embeddings, {
      pineconeIndex: index,
      textKey: "text",
    });

    // // Create MultiQueryRetriever
    // this.retriever = await MultiQueryRetriever.fromLLM({
    //   llm: this.llm,
    //   retriever: this.vectorStore.asRetriever({
    //     searchType: "mmr",
    //     filter: { userId: userId },
    //     searchKwargs: {
    //       fetchK: 100,
    //       lambda: 0.1,
    //     },
    //   }),
    //   queryCount: 10,
    //   prompt: this.queryGenerationPrompt,
    //   verbose: true,
    // });

    this.retriever = this.vectorStore.asRetriever({
      filter: { userId: userId },
      k: 30, // Number of results to return
    });
  }

  async ingestDocument(text: string, metadata: Record<string, any> = {}) {
    console.log("📥 Starting document ingestion");
    try {
      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 100,
        separators: ["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        keepSeparator: true,
      });

      const docs = await textSplitter.createDocuments([text]);

      // Enhance documents with metadata and create prefixed IDs
      const enhancedDocs = docs.map((doc, index) => {
        // Create a prefixed ID that includes userId: user_123#doc_456#chunk_789
        const prefixedId = `user_${
          metadata.userId
        }#doc_${Date.now()}#chunk_${index}`;

        return new Document({
          pageContent: doc.pageContent,
          metadata: {
            ...doc.metadata,
            userId: metadata.userId,
            question: metadata.question,
            timestamp: Date.now(),
            chunkIndex: index,
            documentId: metadata.documentId || `doc-${Date.now()}`,
            title: metadata.title,
            source: metadata.source,
            id: prefixedId,
          },
        });
      });

      await this.vectorStore.addDocuments(enhancedDocs);

      console.log(`✅ Document successfully ingested (${docs.length} chunks)`);
      return docs.length;
    } catch (error) {
      console.error("❌ Error in document ingestion:", error);
      throw error;
    }
  }

  async similaritySearch(query: string) {
    try {
      console.log("Searching with filter:", { userId: this.userId });

      this.retriever = this.vectorStore.asRetriever({
        filter: { userId: this.userId },
        k: 30,
      });

      const retrievedDocs = await this.retriever.invoke(query);

      // Convert position to score (earlier = higher score)
      const results = retrievedDocs.map((doc, index) => ({
        pageContent: doc.pageContent,
        metadata: {
          ...doc.metadata,
          text: doc.pageContent,
        },
        score: 1 - index / retrievedDocs.length, // Convert position to 0-1 score
      }));

      if (results.length > 0) {
        console.log(
          `📊 Retrieved ${results.length} results, ranked from most to least relevant`
        );
      }

      return results;
    } catch (error) {
      console.error("❌ Error in similarity search:", error);
      throw error;
    }
  }

  async deleteUserDocuments(userId: string) {
    try {
      console.log(`🗑️ Attempting to delete documents for user: ${userId}`);
      const index = this.pineconeClient.Index(process.env.PINECONE_INDEX_NAME!);

      // List all vectors with the user prefix
      const userPrefix = `user_${userId}#`;
      console.log("Listing vectors with prefix:", userPrefix);

      let allIds: string[] = [];
      let paginationToken: string | undefined;

      // Paginate through all results
      do {
        const statsResponse = await index.describeIndexStats();

        // Get matching vector IDs from the stats
        const matchingIds = Object.keys(
          statsResponse.namespaces?.default || {}
        ).filter((id) => id.startsWith(userPrefix));

        allIds = allIds.concat(matchingIds);
        paginationToken = undefined;

        console.log(`Found ${matchingIds.length} vectors`);
      } while (paginationToken);

      if (allIds.length === 0) {
        console.log("No documents found for user:", userId);
        return true;
      }

      console.log(`Found total of ${allIds.length} vectors to delete`);

      // Delete vectors one at a time
      let successCount = 0;
      for (const id of allIds) {
        try {
          await index.deleteOne(id);
          successCount++;
          console.log(`Deleted vector ${successCount}/${allIds.length}`);
        } catch (deleteError) {
          console.error("Error deleting vector:", {
            id,
            error: deleteError,
          });
        }
      }

      console.log(
        `✅ Successfully deleted ${successCount}/${allIds.length} documents for user: ${userId}`
      );
      return true;
    } catch (error) {
      console.error("❌ Error deleting user documents:", {
        error,
        errorMessage: error instanceof Error ? error.message : "Unknown error",
        errorStack: error instanceof Error ? error.stack : undefined,
        userId,
        indexName: process.env.PINECONE_INDEX_NAME,
      });
      throw error;
    }
  }
}

// Create singleton instance
export const langchainService = new LangChainService();
