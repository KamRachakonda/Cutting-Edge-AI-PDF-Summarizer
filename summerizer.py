import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


class PDFSummarizer:
    def __init__(self, groq_api_key=None):
        self.api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError('GROQ_API_KEY not found')

        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name='openai/gpt-oss-20b',
            temperature=0
        )

    def summarize(self, chunks, summary_type='concise', custom_prompt=''):
        text = '\n\n'.join(chunks)

        if summary_type == 'concise':
            instruction = 'Write a concise summary in 5-10 clear sentences covering the main ideas.'
        elif summary_type == 'detailed':
            instruction = 'Write a detailed summary with headings, explanations, and important supporting points.'
        elif summary_type == 'bar_chart':
            instruction = 'Write a concise summary focused on the document statistics shown in the bar chart.'
        elif summary_type == 'pie_chart':
            instruction = 'Write a concise summary focused on the document statistics shown in the pie chart.'
        else:
            instruction = 'Summarize the document into clear bullet points highlighting the key information.'

        if custom_prompt:
            instruction += f'\n\nAdditional instruction: {custom_prompt}'

        prompt = ChatPromptTemplate.from_messages([
            (
                'system',
                'You are an expert PDF summarization assistant. Create accurate, well-structured summaries.'
            ),
            (
                'human',
                '{instruction}\n\nDocument:\n{text}'
            ),
        ])

        chain = prompt | self.llm

        response = chain.invoke({
            'instruction': instruction,
            'text': text,
        })

        return response.content


class SummaryFormatter:
    @staticmethod
    def format_summary(summary: str) -> str:
        return summary.strip()
