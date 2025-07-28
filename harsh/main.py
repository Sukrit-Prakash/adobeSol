#!/usr/bin/env python3
"""
Adobe Hackathon Round 1A - Approach 1: Multi-Feature Rule-Based Analysis
PDF Heading Detection using PyMuPDF with sophisticated heuristics
"""

import os
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional
import re
import pymupdf as fitz  # PyMuPDF

class PDFHeadingExtractor:
    def __init__(self):
        self.title = ""
        self.outline = []
        self.font_stats = {}
        self.base_font_size = 0
        self.base_font_name = ""
        
    def extract_font_statistics(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract font statistics from the entire document"""
        font_sizes = []
        font_names = []
        font_combinations = []
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_size = span["size"]
                            font_name = span["font"]
                            font_flags = span["flags"]
                            
                            font_sizes.append(font_size)
                            font_names.append(font_name)
                            font_combinations.append(f"{font_name}_{font_size}")
        
        # Calculate statistics
        font_size_counter = Counter(font_sizes)
        font_name_counter = Counter(font_names)
        font_combo_counter = Counter(font_combinations)
        
        # Determine base (most common) font
        most_common_combo = font_combo_counter.most_common(1)[0][0]
        base_font_name, base_font_size = most_common_combo.rsplit('_', 1)
        
        return {
            'font_sizes': font_size_counter,
            'font_names': font_name_counter,
            'font_combinations': font_combo_counter,
            'base_font_size': float(base_font_size),
            'base_font_name': base_font_name,
            'avg_font_size': sum(font_sizes) / len(font_sizes) if font_sizes else 12
        }
    
    def is_bold_font(self, font_name: str, font_flags: int) -> bool:
        """Determine if text is bold based on font name and flags"""
        # Check font flags (bit 4 is bold)
        if font_flags & (1 << 4):
            return True
        
        # Check font name patterns
        bold_indicators = ['bold', 'Black', 'Heavy', 'Medium', 'Semi', 'Demi']
        font_lower = font_name.lower()
        
        return any(indicator.lower() in font_lower for indicator in bold_indicators)
    
    def calculate_line_metrics(self, line: Dict) -> Dict[str, float]:
        """Calculate metrics for a text line"""
        if not line["spans"]:
            return {}
        
        # Get bounding box
        bbox = line["bbox"]
        line_height = bbox[3] - bbox[1]
        line_width = bbox[2] - bbox[0]
        
        # Get text content
        text = "".join(span["text"] for span in line["spans"])
        
        # Calculate average font size for the line
        total_chars = sum(len(span["text"]) for span in line["spans"])
        weighted_font_size = sum(
            span["size"] * len(span["text"]) for span in line["spans"]
        ) / max(total_chars, 1)
        
        return {
            'height': line_height,
            'width': line_width,
            'text': text.strip(),
            'avg_font_size': weighted_font_size,
            'char_count': len(text.strip()),
            'word_count': len(text.strip().split()),
            'x_position': bbox[0],
            'y_position': bbox[1]
        }
    
    def classify_heading_level(self, font_size: float, is_bold: bool, 
                             base_font_size: float, text: str) -> Optional[str]:
        """Classify heading level based on font characteristics"""
        size_ratio = font_size / base_font_size if base_font_size > 0 else 1
        
        # Title detection (usually largest, often on first pages)
        if size_ratio > 1.8 or (size_ratio > 1.5 and len(text.split()) <= 10):
            return "TITLE"
        
        # H1: Significantly larger than base or bold with good size increase
        if size_ratio > 1.4 or (is_bold and size_ratio > 1.2):
            return "H1"
        
        # H2: Moderately larger than base or bold with some size increase
        if size_ratio > 1.15 or (is_bold and size_ratio > 1.0):
            return "H2"
        
        # H3: Slightly larger than base or bold at base size
        if size_ratio > 1.05 or (is_bold and size_ratio >= 0.95):
            return "H3"
        
        return None
    
    def is_likely_heading(self, text: str, font_size: float, is_bold: bool,
                         base_font_size: float, y_position: float,
                         prev_y: float, line_height: float) -> bool:
        """Additional heuristics to determine if text is a heading"""
        
        # Skip very short or very long text
        if len(text) < 3 or len(text) > 200:
            return False
        
        # Skip text that looks like page numbers, footnotes, etc.
        if re.match(r'^\d+$', text.strip()):
            return False
        
        if re.match(r'^[ivxlcdm]+\.?\s*$', text.strip().lower()):
            return False
        
        # Check for heading patterns
        heading_patterns = [
            r'^\d+\.?\s+',  # "1. Introduction"
            r'^\d+\.\d+\.?\s+',  # "1.1. Overview"
            r'^[A-Z][A-Z\s]{2,}$',  # "INTRODUCTION"
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]*)*$',  # "Introduction"
        ]
        
        pattern_match = any(re.match(pattern, text) for pattern in heading_patterns)
        
        # Check spacing (headings often have more vertical space)
        has_spacing = abs(y_position - prev_y) > line_height * 1.5 if prev_y > 0 else True
        
        # Font size or bold requirement
        size_check = font_size > base_font_size * 1.05 or is_bold
        
        return (pattern_match or has_spacing) and size_check
    
    def extract_title(self, doc: fitz.Document) -> str:
        """Extract document title from first few pages"""
        candidates = []
        
        # Check first 3 pages for title
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") == 0:
                    for line in block["lines"]:
                        metrics = self.calculate_line_metrics(line)
                        if not metrics or metrics['char_count'] == 0:
                            continue
                        
                        # Check if this could be a title
                        spans = line["spans"]
                        if spans:
                            font_size = spans[0]["size"]
                            font_flags = spans[0]["flags"]
                            is_bold = self.is_bold_font(spans[0]["font"], font_flags)
                            
                            # Title criteria
                            if (font_size > self.base_font_size * 1.3 and 
                                5 <= metrics['word_count'] <= 15 and
                                metrics['y_position'] < page.rect.height * 0.4):
                                
                                candidates.append({
                                    'text': metrics['text'],
                                    'font_size': font_size,
                                    'page': page_num,
                                    'y_pos': metrics['y_position'],
                                    'score': font_size + (10 if is_bold else 0) - metrics['y_position'] * 0.01
                                })
        
        if candidates:
            # Return highest scoring candidate
            best_candidate = max(candidates, key=lambda x: x['score'])
            return best_candidate['text']
        
        return "Untitled Document"
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Main processing function"""
        try:
            doc = fitz.open(pdf_path)
            
            # Extract font statistics
            self.font_stats = self.extract_font_statistics(doc)
            self.base_font_size = self.font_stats['base_font_size']
            self.base_font_name = self.font_stats['base_font_name']
            
            print(f"Base font: {self.base_font_name}, size: {self.base_font_size}")
            
            # Extract title
            self.title = self.extract_title(doc)
            
            # Extract headings
            self.outline = []
            prev_y_positions = {}
            
            for page_num, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]
                prev_y = prev_y_positions.get(page_num, 0)
                
                for block in blocks:
                    if block.get("type") == 0:  # Text block
                        for line in block["lines"]:
                            metrics = self.calculate_line_metrics(line)
                            if not metrics or metrics['char_count'] == 0:
                                continue
                            
                            spans = line["spans"]
                            if not spans:
                                continue
                            
                            # Get font characteristics
                            span = spans[0]  # Use first span characteristics
                            font_size = span["size"]
                            font_flags = span["flags"]
                            is_bold = self.is_bold_font(span["font"], font_flags)
                            
                            # Check if this is a heading
                            if self.is_likely_heading(
                                metrics['text'], font_size, is_bold,
                                self.base_font_size, metrics['y_position'],
                                prev_y, metrics['height']
                            ):
                                level = self.classify_heading_level(
                                    font_size, is_bold, self.base_font_size, metrics['text']
                                )
                                
                                if level and level != "TITLE":
                                    self.outline.append({
                                        "level": level,
                                        "text": metrics['text'],
                                        "page": page_num + 1  # 1-indexed
                                    })
                            
                            prev_y = metrics['y_position']
                            prev_y_positions[page_num] = prev_y
            
            doc.close()
            
            return {
                "title": self.title,
                "outline": self.outline
            }
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return {
                "title": "Error Processing Document",
                "outline": []
            }

def main():
    """Main execution function"""
    # input_dir = Path("/input")
    # output_dir = Path("/output")
    project_root = Path(__file__).parent.resolve()
    input_dir = project_root / "input"
    output_dir = project_root / "output"
    
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Process all PDF files in input directory
    for pdf_file in input_dir.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        
        extractor = PDFHeadingExtractor()
        result = extractor.process_pdf(str(pdf_file))
        
        # Save result
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Saved: {output_file.name}")
        print(f"Title: {result['title']}")
        print(f"Headings found: {len(result['outline'])}")

if __name__ == "__main__":
    main()
