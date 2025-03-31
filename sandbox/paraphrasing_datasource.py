from abc import ABC, abstractmethod
from typing import Dict, List, Iterator
import json
import pandas as pd

class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    def get_phrases(self) -> Iterator[Dict[str, str]]:
        """
        Returns an iterator of phrases with metadata
        
        Each item should be a dict with at least:
        - 'text': the phrase to paraphrase
        - 'source': identifier for the data source
        - Additional metadata specific to the source
        """
        pass

class TestPhrasesDataSource(DataSource):
    """Simple test phrases data source"""
    
    def __init__(self, phrases: List[str]):
        self.phrases = phrases
        
    def get_phrases(self) -> Iterator[Dict[str, str]]:
        for i, phrase in enumerate(self.phrases):
            yield {
                'text': phrase,
                'source': 'test_phrases',
                'phrase_id': i
            }

class LeetCodeDataSource(DataSource):
    """LeetCode dataset data source"""
    
    def __init__(self, file_path: str):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
            
    def get_phrases(self) -> Iterator[Dict[str, str]]:
        for entry in self.data:
            yield {
                'text': entry['question'],
                'source': 'leetcode',
                'phrase_id': entry['number']
            }
        
class TasksDataSetDataSource(DataSource):
    """Tasks dataset data source"""
    
    def __init__(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def get_phrases(self) -> Iterator[Dict[str, str]]:
        for entry in self.data:
            yield {
                'text': entry['question'],
                'source': 'tasks dataset',
                'phrase_id': entry['id']
            }

class CSVDataSource(DataSource):
    """CSV file data source"""
    
    def __init__(self, file_path: str, text_column: str):
        self.df = pd.read_csv(file_path)
        self.text_column = text_column
        
    def get_phrases(self) -> Iterator[Dict[str, str]]:
        for _, row in self.df.iterrows():
            yield {
                'text': row[self.text_column],
                'source': 'csv',
                'phrase_id': _
            } 