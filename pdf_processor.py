import os
import re
import time
from typing import Dict, List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import pypdf
import tiktoken


class PDFProcessor:
    """Reusable PDF extraction and document inspection helpers."""

    def __init__(self):
        self.encoding = tiktoken.get_encoding('cl100k_base')

    def extract_text_from_pdf(self, pdf_file) -> str:
        reader = pypdf.PdfReader(pdf_file)
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            pages.append(
                f'\n--- Page {page_number} ---\n{page.extract_text() or ""}')
        return ''.join(pages)

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'--- Page \d+ ---', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    @staticmethod
    def get_pdf_metadata(pdf_file) -> Dict:
        try:
            reader = pypdf.PdfReader(pdf_file)
            metadata = reader.metadata or {}
            return {
                'num_pages': len(reader.pages),
                'title': metadata.get('/Title', 'Unknown'),
                'author': metadata.get('/Author', 'Unknown'),
                'subject': metadata.get('/Subject', 'Unknown'),
            }
        except Exception as error:
            return {'error': str(error)}


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

    @staticmethod
    def chunk_text(text: str, max_characters: int = 12000) -> List[str]:
        """Split long documents at paragraph or sentence boundaries."""
        if len(text) <= max_characters:
            return [text]

        chunks = []
        remaining = text.strip()
        while remaining:
            if len(remaining) <= max_characters:
                chunks.append(remaining)
                break
            boundary = remaining.rfind('\n\n', 0, max_characters)
            if boundary < max_characters // 2:
                boundary = remaining.rfind('. ', 0, max_characters)
            if boundary < max_characters // 2:
                boundary = max_characters
            chunks.append(remaining[:boundary].strip())
            remaining = remaining[boundary:].strip()
        return [chunk for chunk in chunks if chunk]

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

    def summarize_chunks(self, chunks: List[str], summary_type: str = 'detailed',
                         custom_prompt: str = '') -> Dict:
        """Summarize each chunk and synthesize the successful results."""
        chunk_summaries = []
        for index, chunk in enumerate(chunks):
            try:
                summary = self.summarize(
                    [chunk], summary_type=summary_type, custom_prompt=custom_prompt)
                chunk_summaries.append({
                    'chunk_number': index + 1,
                    'summary': summary,
                    'original_length': len(chunk),
                    'summary_length': len(summary),
                })
            except Exception as error:
                chunk_summaries.append({
                    'chunk_number': index + 1,
                    'summary': f'Error processing chunk: {error}',
                    'original_length': len(chunk),
                    'summary_length': 0,
                })
            if index < len(chunks) - 1:
                time.sleep(1)

        return {
            'individual_summaries': chunk_summaries,
            'combined_summary': self.combine_summaries(chunk_summaries, summary_type),
            'total_chunks': len(chunks),
            'summary_type': summary_type,
        }

    def combine_summaries(self, chunk_summaries: List[Dict], summary_type: str) -> str:
        """Create one cohesive summary from successful chunk summaries."""
        summaries = [
            item['summary'] for item in chunk_summaries
            if not item['summary'].startswith('Error')
        ]
        if not summaries:
            return 'No valid summaries were generated.'

        prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are an expert editor synthesizing document summaries.'),
            ('human',
             'Create a cohesive {summary_type} summary from these section summaries.\n\n'
             '{summaries}\n\nFinal summary:')
        ])
        response = (prompt | self.llm).invoke({
            'summary_type': summary_type,
            'summaries': '\n\n'.join(
                f'Section {index + 1}: {summary}'
                for index, summary in enumerate(summaries)
            ),
        })
        return response.content

    def analyze_document_structure(self, text: str) -> Dict[str, str]:
        """Identify document type, themes, structure, audience, and purpose."""
        prompt = ChatPromptTemplate.from_messages([
            ('system', 'You analyze documents clearly and accurately.'),
            ('human',
             'Analyze this document and provide its type, main themes, key sections, '
             'target audience, and overall purpose.\n\n{text}')
        ])
        try:
            response = (prompt | self.llm).invoke({'text': text[:12000]})
            return {'analysis': response.content, 'status': 'success'}
        except Exception as error:
            return {'analysis': f'Error analyzing document: {error}', 'status': 'error'}

    def extract_key_quotes(self, text: str) -> List[str]:
        """Extract up to ten notable statements from the document."""
        prompt = ChatPromptTemplate.from_messages([
            ('system', 'You select faithful, verbatim quotes from documents.'),
            ('human',
             'Extract 5-10 key quotes or important phrases from this text. Return '
             'one quote per line and do not add commentary.\n\n{text}')
        ])
        try:
            response = (prompt | self.llm).invoke({'text': text[:30000]})
            return [line.strip('- *\t ') for line in response.content.splitlines()
                    if line.strip()][:10]
        except Exception as error:
            return [f'Error extracting quotes: {error}']


class SummaryFormatter:
    @staticmethod
    def format_summary(summary: str) -> str:
        return summary.strip()
