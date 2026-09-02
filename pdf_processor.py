import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import pypdf


class PDFSummarizer:
    def __init__(self, groq_api_key=None):
        self.api_key = (groq_api_key or os.getenv('GROQ_API_KEY') or '').strip(
            " \'\"\u2018\u2019\u201c\u201d")
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
            instruction = 'Write a concise summary in 5-10 clear sentences.'
        elif summary_type == 'detailed':
            instruction = 'Write a detailed summary with headings and key points.'
        elif summary_type == 'bar_chart':
            instruction = 'Write a concise summary focused on the document statistics shown in the bar chart.'
        elif summary_type == 'pie_chart':
            instruction = 'Write a concise summary focused on the document statistics shown in the pie chart.'
        else:
            instruction = 'Summarize the document into clear bullet points.'

        if custom_prompt:
            instruction += f'\n\nAdditional instruction: {custom_prompt}'

        prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are an expert PDF summarizer.'),
            ('human', '{instruction}\n\nDocument:\n{text}')
        ])

        chain = prompt | self.llm

        response = chain.invoke({
            'instruction': instruction,
            'text': text
        })

        return response.content


class SummaryFormatter:
    @staticmethod
    def format_summary(summary: str) -> str:
        return summary.strip()
