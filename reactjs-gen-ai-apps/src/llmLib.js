import { fetchAuthSession } from 'aws-amplify/auth';
import { Bedrock } from "@langchain/community/llms/bedrock/web";
import { AmazonKnowledgeBaseRetriever } from "@langchain/community/retrievers/amazon_knowledge_base";
import { ConversationChain, ConversationalRetrievalQAChain } from "langchain/chains";
import { BedrockAgentClient, ListKnowledgeBasesCommand } from "@aws-sdk/client-bedrock-agent"
import { BedrockAgentRuntimeClient, RetrieveAndGenerateCommand, RetrieveCommand } from "@aws-sdk/client-bedrock-agent-runtime"



export const getModel = async () => {
    const session = await fetchAuthSession(); //To get user credential from React
    const model = new Bedrock({
        model: "anthropic.claude-v2:1",
        region: "us-east-1",
        streaming: true,
        credentials: session.credentials,
        modelKwargs: {max_tokens_to_sample: 4096, temperature: 0.8, top_p: 0.32, top_k: 175, stop_sequences: []},
    });
    return model;
};


export const getChain = (llm, memory) => {

    const chain = new ConversationChain({ llm: llm, memory: memory })
    chain.prompt.template = `The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. 
Current conversation:
{history}

Human: {input}
Assistant:`
    return chain
}


export const getBedrockKnowledgeBases = async () => {
    const session = await fetchAuthSession()
    const client = new BedrockAgentClient({ region: "us-east-1", credentials: session.credentials })
    const command = new ListKnowledgeBasesCommand({})
    const response = await client.send(command)
    return response.knowledgeBaseSummaries
}


export const ragBedrockKnowledgeBase = async (sessionId, knowledgeBaseId, query) => {
    const session = await fetchAuthSession()
    const client = new BedrockAgentRuntimeClient({ region: "us-east-1", credentials: session.credentials });
    const input = {
        input: { text: query },
        retrieveAndGenerateConfiguration: {
            type: "KNOWLEDGE_BASE",
            knowledgeBaseConfiguration: {
                knowledgeBaseId: knowledgeBaseId,
                modelArn: "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2:1", // required
            },
        }
    }

    if (sessionId) {
        input.sessionId = sessionId
    }

    const command = new RetrieveAndGenerateCommand(input);
    const response = await client.send(command)
    return response
}



export const retrieveBedrockKnowledgeBase = async (knowledgeBaseId, query) => {
    const session = await fetchAuthSession()
    const client = new BedrockAgentRuntimeClient({ region: "us-east-1", credentials: session.credentials });
    const input = { // RetrieveRequest
        knowledgeBaseId: knowledgeBaseId, // required
        retrievalQuery: { // KnowledgeBaseQuery
            text: query, // required
        },
        retrievalConfiguration: { // KnowledgeBaseRetrievalConfiguration
            vectorSearchConfiguration: { // KnowledgeBaseVectorSearchConfiguration
                numberOfResults: 5, // required
            },
        }
    }


    const command = new RetrieveCommand(input);
    const response = await client.send(command)
    return response
}


export const getBedrockKnowledgeBaseRetriever = async (knowledgeBaseId) => {
    const session = await fetchAuthSession();

    const retriever = new AmazonKnowledgeBaseRetriever({
        topK: 10,
        knowledgeBaseId: knowledgeBaseId,
        region: "us-east-1",
        clientOptions: { credentials: session.credentials }
    })

    return retriever

}


export const getConversationalRetrievalQAChain = async (llm, retriever, memory) => {


    const chain = ConversationalRetrievalQAChain.fromLLM(
        llm, retriever = retriever)
    chain.memory = memory

    chain.questionGeneratorChain.prompt.template = "Human: " + chain.questionGeneratorChain.prompt.template +"\nAssistant:"

    chain.combineDocumentsChain.llmChain.prompt.template = 
`Human:
This is a conversation between human and a virtual assistant.
Your persona is a virtual assistant created by Bina Nusantara university. Your name is Claudia.
If human greets you, ignore the context below, greets back and introduce yourself.
If human asks you question related to Bina Nusantara, Binus, university, students, or staffs, try to answer based on provided context.

Answer based on context below
<context>
{context}
</context>

This is the chat history
<chat_history>
{chat_history}
</chat_history>


If you did not get the answer from the context above, just say that you don't know and refer human to Binus Support team.
Don't try to make up an answer.
Be concise but friendly in your answer and do not mention you get it from the context above.
Respond only in Bahasa Indonesia.

Here is the question that human asks.
<question>
{question}
</question>

Assistant:`
/*        `Human: Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 

{context}

Question: {question}
Helpful Answer:
Assistant:`*/

return chain
}


