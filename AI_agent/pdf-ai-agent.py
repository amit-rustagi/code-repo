# pdf_agent/document_processor.py
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
from transformers import pipeline
import spacy
from typing import List, Dict, Any, Optional
import io
import os

class PDFProcessor:
    def __init__(self):
        # Initialize NLP components
        self.nlp = spacy.load("en_core_web_sm")
        self.qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
    def load_pdf(self, file_path: str) -> fitz.Document:
        """Load a PDF file and return the document object."""
        return fitz.open(file_path)
    
    def extract_text(self, doc: fitz.Document) -> str:
        """Extract text from all pages of the PDF."""
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    def extract_images(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Extract images from the PDF with their locations."""
        images = []
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                if base_image:
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_dict = {
                        "page": page_num + 1,
                        "index": img_index,
                        "extension": image_ext,
                        "bytes": image_bytes
                    }
                    images.append(image_dict)
        return images
    
    def ocr_image(self, image_bytes: bytes) -> str:
        """Perform OCR on an image."""
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image)
    
    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Perform NLP analysis on the document text."""
        doc = self.nlp(text)
        
        # Extract entities
        entities = {ent.label_: [] for ent in doc.ents}
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
            
        # Extract key phrases
        key_phrases = []
        for chunk in doc.noun_chunks:
            if chunk.root.dep_ in ['nsubj', 'dobj', 'pobj']:
                key_phrases.append(chunk.text)
                
        return {
            "entities": entities,
            "key_phrases": key_phrases,
            "sentiment": self._analyze_sentiment(text),
            "summary": self._generate_summary(text)
        }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of the text."""
        sentiment_analyzer = pipeline("sentiment-analysis")
        result = sentiment_analyzer(text[:512])[0]  # Process first 512 chars for efficiency
        return {
            "label": result["label"],
            "score": result["score"]
        }
    
    def _generate_summary(self, text: str) -> str:
        """Generate a summary of the text."""
        # Split text into chunks if it's too long
        max_chunk_length = 1024
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        summaries = []
        for chunk in chunks:
            summary = self.summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        return " ".join(summaries)
    
    def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """Answer a question based on the document content."""
        result = self.qa_pipeline(question=question, context=context)
        return {
            "answer": result["answer"],
            "confidence": result["score"],
            "start": result["start"],
            "end": result["end"]
        }
    
    def search_document(self, query: str, text: str) -> List[Dict[str, Any]]:
        """Search for relevant sections in the document."""
        doc = self.nlp(text)
        query_doc = self.nlp(query)
        
        # Split into paragraphs and calculate similarity
        paragraphs = [p.text.strip() for p in doc.sents if len(p.text.strip()) > 0]
        results = []
        
        for para in paragraphs:
            para_doc = self.nlp(para)
            similarity = para_doc.similarity(query_doc)
            if similarity > 0.5:  # Adjustable threshold
                results.append({
                    "text": para,
                    "similarity": similarity
                })
        
        return sorted(results, key=lambda x: x["similarity"], reverse=True)

# pdf_agent/agent.py
class PDFAgent:
    def __init__(self):
        self.processor = PDFProcessor()
        self.current_document = None
        self.extracted_text = ""
        self.analysis_results = None
        
    def load_document(self, file_path: str) -> bool:
        """Load a PDF document."""
        try:
            self.current_document = self.processor.load_pdf(file_path)
            self.extracted_text = self.processor.extract_text(self.current_document)
            return True
        except Exception as e:
            print(f"Error loading document: {str(e)}")
            return False
    
    def analyze_document(self) -> Dict[str, Any]:
        """Perform comprehensive document analysis."""
        if not self.extracted_text:
            raise ValueError("No document loaded")
        
        self.analysis_results = self.processor.analyze_document(self.extracted_text)
        return self.analysis_results
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Answer a question about the document."""
        if not self.extracted_text:
            raise ValueError("No document loaded")
            
        return self.processor.answer_question(question, self.extracted_text)
    
    def search_content(self, query: str) -> List[Dict[str, Any]]:
        """Search for relevant content in the document."""
        if not self.extracted_text:
            raise ValueError("No document loaded")
            
        return self.processor.search_document(query, self.extracted_text)
    
    def extract_images(self) -> List[Dict[str, Any]]:
        """Extract images from the document."""
        if not self.current_document:
            raise ValueError("No document loaded")
            
        return self.processor.extract_images(self.current_document)
    
    def get_text_from_image(self, image_bytes: bytes) -> str:
        """Extract text from an image using OCR."""
        return self.processor.ocr_image(image_bytes)
