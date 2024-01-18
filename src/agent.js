import { CSVLoader } from 'langchain/document_loaders/fs/csv'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { MemoryVectorStore } from 'langchain/vectorstores/memory'
import { OpenAI } from 'langchain/llms/openai'
import { PromptTemplate } from 'langchain/prompts'

const knowledgeBase = './data/inquiries.csv'

const loader = new CSVLoader(knowledgeBase)
const data = await loader.load()

const message = 'Can we have a tasting session?'

const vectorStore = await MemoryVectorStore.fromDocuments(
  data,
  new OpenAIEmbeddings()
)

const relevantDocs = await vectorStore.similaritySearch(message, 1)

const promptTemplate = new PromptTemplate({
  inputVariables: ['message', 'best_practice'],
  template: `
  As a seasoned customer support agent, you have a knack for addressing client 
  inquiries based on past successful interactions. 

  Here's a message we received from a client:

  {message}

  For reference, here are some of our historically effective responses to similar questions:
  {best_practice}

  Using this background, please craft a response that aligns with our established standards
  and best addresses the client's concern.
  `
})

const bestPractices = relevantDocs.map((doc) => doc.pageContent)

const query = await promptTemplate.format({
  message,
  best_practice: bestPractices
})

const llm = new OpenAI()
const result = await llm.predict(query)

console.log(result)