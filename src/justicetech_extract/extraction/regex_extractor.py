"""
Regex-based extraction for Franklin County Municipal Court documents.

Full port of ImprovedExtractor from court_document_extractor.py v3.9.
All patterns are identical to the production-validated version.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple, List, Dict, Any

from dateutil import parser

from justicetech_extract.models import (
    ExtractedCourtInfo,
    OutcomeType,
    PaymentScheduleItem,
    PaymentType,
)


def normalize_date(date_str: Optional[str]) -> Optional[str]:
    """Normalize ALL date strings into consistent 'YYYY-MM-DD' format for easy parsing"""
    if not date_str:
        return None
    try:
        date_str = re.sub(r'\s+by\s+\d{1,2}:\d{2}\s*(?:AM|PM)?', '', date_str, flags=re.IGNORECASE)
        date_str = re.sub(r'\s+at\s+\d{1,2}:\d{2}\s*(?:AM|PM)?', '', date_str, flags=re.IGNORECASE)
        date_str = date_str.strip()
        dt = parser.parse(date_str, fuzzy=True, dayfirst=False)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        try:
            match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', date_str)
            if match:
                month, day, year = match.groups()
                if len(year) == 2:
                    year = '20' + year
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        except:
            pass
        return date_str


class RegexExtractor:
    """Improved regex-based extraction tailored for Franklin County documents"""
    
    @staticmethod
    def extract_case_number(text: str, filename: Optional[str] = None) -> Optional[str]:
        """
        Extract case number from filename or text.
        
        v3.6: Prioritizes structured filename patterns like:
          2024_CVG_000002_m5hbU9VYVRw_2024_CVG_000002_-_1_22_2024_-_DAGREED_-_..._cleaned.txt
          
        The canonical case number comes from the filename prefix (e.g., "2024 CVG 000002")
        rather than the OCR'd text which may be truncated (e.g., "24 CVG 02").
        """
        if filename:
            clean_filename = re.sub(r'\.(txt|png|pdf|jpg|jpeg)$', '', filename, flags=re.IGNORECASE)
            clean_filename = re.sub(r'_png$', '', clean_filename)
            clean_filename = re.sub(r'_cleaned$', '', clean_filename)
            
            # Priority 1: Structured filename prefix "YYYY_CVG_NNNNNN_..."
            match = re.search(r'^(\d{4})_CVG_(\d{5,6})', clean_filename)
            if match:
                return f"{match.group(1)} CVG {match.group(2)}"
            
            # Priority 2: Explicit "YYYY CVG NNNNNN" with spaces anywhere in filename
            match = re.search(r'(\d{4})\s+CVG\s+(\d{5,6})', clean_filename, re.IGNORECASE)
            if match:
                return f"{match.group(1)} CVG {match.group(2)}"
            
            # Priority 3: Underscore-separated "YYYY_CVG_NNNNNN" anywhere
            match = re.search(r'(\d{4})[_]CVG[_](\d{5,6})', clean_filename, re.IGNORECASE)
            if match:
                return f"{match.group(1)} CVG {match.group(2)}"
            
            # Priority 4: Other separators or concatenated forms
            patterns = [
                r'(\d{2,4})[_\s-]+(CVG)[_\s-]+(\d{5,6})',
                r'(\d{4})(CVG)(\d{6})',
                r'(\d{2})(CVG)(\d{5,6})',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, clean_filename, re.IGNORECASE)
                if match:
                    year = match.group(1)
                    if len(year) == 2:
                        year = '20' + year
                    return f"{year} CVG {match.group(3)}"
        
        # Fallback: extract from document text (may be truncated by OCR)
        patterns = [
            r'(?:Case\s+No\.?|CASE\s+NO\.?)[:\s]*(\d{4}\s*CVG\s*\d{6})',
            r'(?:Case\s+No\.?|CASE\s+NO\.?)[:\s]*(\d{2}[_\s]CVG[_\s]\d{5,6})',
            r'(\d{4}\s*CVG\s*\d{6})',
            r'(\d{2}[_\s]CVG[_\s]\d{5,6})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                case_num = match.group(1)
                case_num = re.sub(r'[_-]', ' ', case_num)
                case_num = re.sub(r'\s+', ' ', case_num)
                parts = case_num.split()
                if len(parts) == 3 and len(parts[0]) == 2:
                    parts[0] = '20' + parts[0]
                return ' '.join(parts).strip()
        
        return None
    
    @staticmethod
    def extract_agreement_date(text: str, filename: Optional[str] = None) -> Optional[str]:
        """
        Extract agreement signed date from filename or text.
        
        v3.6: Prioritizes structured filename patterns like:
          ..._-_1_22_2024_-_DAGREED_-_CV_Docket_-_1_23_2024_cleaned.txt
          
        The first date (before DAGREED) is the agreement/case date.
        The second date (after CV Docket) is the docket filing date.
        """
        if filename:
            clean_filename = re.sub(r'\.(txt|png|pdf|jpg|jpeg)$', '', filename, flags=re.IGNORECASE)
            clean_filename = re.sub(r'_cleaned$', '', clean_filename)
            
            # Priority 1: Structured filename with _-_{M}_{D}_{YYYY}_-_DAGREED
            match = re.search(
                r'_-_(\d{1,2})_(\d{1,2})_(\d{4})_-_DAGREED',
                clean_filename, re.IGNORECASE
            )
            if match:
                month = match.group(1).zfill(2)
                day = match.group(2).zfill(2)
                year = match.group(3)
                return f"{year}-{month}-{day}"
            
            # Priority 2: Structured filename with spaces instead of underscores
            match = re.search(
                r'\s*-\s*(\d{1,2})\s+(\d{1,2})\s+(\d{4})\s*-\s*DAGREED',
                clean_filename, re.IGNORECASE
            )
            if match:
                month = match.group(1).zfill(2)
                day = match.group(2).zfill(2)
                year = match.group(3)
                return f"{year}-{month}-{day}"
        
        # Fallback: extract from document text
        judge_patterns = [
            r'(?:JUDGE|MAGISTRATE)[\s\S]{0,100}?(\d{1,2}/\d{1,2}/\d{2,4})',
            r'(?:JUDGE|MAGISTRATE)[\s\S]{0,100}?((?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{1,2}[,\s]+\d{4})',
            r'Date[\s:]*<?signature>?[\s\S]{0,50}?(\d{1,2}/\d{1,2}/\d{2,4})',
            r'Date[\s:]*<?signature>?[\s\S]{0,50}?((?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{1,2}[,\s]+\d{4})',
        ]
        
        for pattern in judge_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                date_str = match.group(1).strip()
                if re.search(r'\d', date_str):
                    return normalize_date(date_str)
        
        watermark_patterns = [
            r'<watermark>.*?(\d{4}\s+[A-Z]{3}\s+\d{1,2}[\s,]+(?:AM|PM)\s+\d{1,2}:\d{2}).*?</watermark>',
            r'<watermark>.*?([A-Z]{3}\s+\d{1,2}\s+\d{4}).*?</watermark>',
        ]
        
        for pattern in watermark_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()
                if re.search(r'\d{1,2}', date_str):
                    return normalize_date(date_str)
        
        stamp_pattern = r'(\d{4}\s+[A-Z]{3}\s+\d{1,2})(?:\s+[AP]M?)?'
        match = re.search(stamp_pattern, text)
        if match:
            return normalize_date(match.group(1).strip())
        
        return None
    
    @staticmethod
    def extract_parties(text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract plaintiff and defendant from Franklin County documents.
        
        v3.9: Added Method 0 for table-format party blocks:
        - Layout A: Name in <td>, "Plaintiff," in next <tr> row
        - Layout B: "Name<br>Plaintiff," in single <td> cell
        - Added single-character signature filter
        
        v3.7: Handles multiple document formats:
        - Standard: Name\\nPlaintiff,\\nv.\\nName\\nDefendant
        - Signature-only: <signature>Name</signature>\\nPlaintiff,
        - Inline: PLAINTIFF(S), ... v. ... DEFENDANT(S)
        - Same-line: "v. Name" 
        - Multi-line: "Company Inc /\\nSubsidiary, Plaintiff,"
        """
        plaintiff = None
        defendant = None
        
        # Remove page markers and watermarks for clean matching
        clean_text = re.sub(r'<page_number>.*?</page_number>', '', text)
        clean_text = re.sub(r'<watermark>.*?</watermark>', '', clean_text)
        clean_text = re.sub(r'<img>.*?</img>', '', clean_text)
        clean_text = re.sub(r'^\s*[mM]\d+\s*$', '', clean_text, flags=re.MULTILINE)
        clean_text = re.sub(r'^=+\s*PAGE\s+\d+\s*=+\s*$', '', clean_text, flags=re.MULTILINE)
        
        # Strategy 1: Extract names from <signature> tags near Plaintiff/Defendant labels
        sig_names = re.findall(r'<signature>(.*?)</signature>', clean_text)
        
        # Remove signature tags to get clean text for regex matching
        text_no_sigs = re.sub(r'<signature>.*?</signature>', '', clean_text)
        text_no_sigs = re.sub(r'</?u>', '', text_no_sigs)
        
        # ============================================================
        # v3.9: METHOD 0 — Extract from table-formatted party blocks
        # Handles two common layouts:
        #   Layout A: <td>Hearty Home</td> ... <td>Plaintiff,</td> (separate rows)
        #   Layout B: <td>Blueprint<br>Plaintiff,</td> (name+label in one cell)
        # Only examines first 2 tables (party info is always near top)
        # ============================================================
        tables = re.findall(r'<table>(.*?)</table>', clean_text, re.DOTALL)
        for table_content in tables[:2]:
            rows = re.findall(r'<tr>(.*?)</tr>', table_content, re.DOTALL)
            
            # Collect ALL cells in order for Layout A scanning
            # v3.9: Strip signature tags from cells to avoid picking up sig content as names
            all_cells = []
            for row in rows:
                cells = re.findall(r'<td>(.*?)</td>', row, re.DOTALL)
                for c in cells:
                    cleaned_cell = re.sub(r'<signature>.*?</signature>', '', c).strip()
                    all_cells.append(cleaned_cell)
            
            # --- Layout B: "Name<br>Plaintiff," or "Name<br>Defendant," in single cell ---
            for cell in all_cells:
                if '<br>' in cell:
                    # v3.9: Also strip any remaining sig tags
                    cell_clean = re.sub(r'<signature>.*?</signature>', '', cell)
                    parts = cell_clean.split('<br>')
                    if len(parts) >= 2:
                        name_part = parts[0].strip()
                        label_part = parts[1].strip().rstrip(',.:')
                        label_lower = label_part.lower()
                        
                        if not name_part or len(name_part) < 2:
                            continue
                        # Skip if name looks like a header
                        if re.search(r'MUNICIPAL COURT|FRANKLIN COUNTY|^IN THE', name_part, re.IGNORECASE):
                            continue
                        
                        if 'plaintiff' in label_lower and not plaintiff:
                            plaintiff = name_part
                        elif 'defendant' in label_lower and not defendant:
                            defendant = name_part
            
            # --- Layout A: Name in cell before a cell containing "Plaintiff," ---
            if not plaintiff or not defendant:
                for i, cell in enumerate(all_cells):
                    cell_clean = cell.strip().rstrip(',.:')
                    cell_lower = cell_clean.lower()
                    
                    if cell_lower.startswith('plaintiff') and not plaintiff:
                        # Walk backwards to find the name cell, skipping ":", empty, case-number cells
                        for j in range(i - 1, -1, -1):
                            candidate = all_cells[j].strip()
                            cand_lower = candidate.lower().rstrip(',.:')
                            if (candidate and candidate != ':' and len(candidate) > 1
                                    and cand_lower not in ('v.', 'vs.', 'defendant', 'defendants', 'plaintiff', 'plaintiffs', 'date')
                                    and not candidate.upper().startswith('CASE')
                                    and not re.search(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$', candidate)
                                    and not re.search(r'MUNICIPAL COURT|FRANKLIN COUNTY|^IN THE', candidate, re.IGNORECASE)):
                                plaintiff = candidate
                                break
                    
                    elif cell_lower.startswith('defendant') and not defendant:
                        for j in range(i - 1, -1, -1):
                            candidate = all_cells[j].strip()
                            cand_lower = candidate.lower().rstrip(',.:')
                            if (candidate and candidate != ':' and len(candidate) > 1
                                    and cand_lower not in ('v.', 'vs.', 'defendant', 'defendants', 'plaintiff', 'plaintiffs', 'date')
                                    and not candidate.upper().startswith('CASE')
                                    and not re.search(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$', candidate)
                                    and not re.search(r'^Date$', candidate, re.IGNORECASE)):
                                defendant = candidate
                                break
        
        # === PLAINTIFF EXTRACTION (text-based methods) ===
        
        # Method 1: Name on line(s) before "Plaintiff" (standard format)
        if not plaintiff:
            plaintiff_match = re.search(
                r'([A-Z][^\n]+(?:\n[A-Z][^\n]+)*?)\s*[,]?\s*\n\s*(?:Plaintiff|PLAINTIFF)',
                text_no_sigs, re.MULTILINE
            )
            if plaintiff_match:
                candidate = plaintiff_match.group(1).strip()
                candidate = re.sub(r'\s+', ' ', candidate)
                # Skip court headers
                if not re.search(r'MUNICIPAL COURT|FRANKLIN COUNTY|COLUMBUS.*OHIO|^IN THE', candidate, re.IGNORECASE):
                    plaintiff = candidate
        
        # Method 2: Multi-line with "Plaintiff," on same line
        if not plaintiff:
            multi_plaintiff = re.search(
                r'((?:[A-Z][^\n]*\n){0,3}[^\n]*?),?\s*(?:Plaintiff|PLAINTIFF)\s*[,(.]',
                text_no_sigs, re.IGNORECASE
            )
            if multi_plaintiff:
                candidate = multi_plaintiff.group(1).strip()
                candidate = re.sub(r'\n', ' ', candidate)
                candidate = re.sub(r'\s+', ' ', candidate)
                if not re.search(r'MUNICIPAL COURT|FRANKLIN COUNTY|^IN THE', candidate, re.IGNORECASE):
                    plaintiff = candidate
        
        # Method 3: "PLAINTIFF(S)," format (structured court docs)
        if not plaintiff:
            pltf_match = re.search(
                r'([A-Z][A-Za-z\s.,&/]+?)\s*(?:,\s*)?(?:PLAINTIFF\(S\)|Plaintiff\(s\))',
                text_no_sigs, re.IGNORECASE
            )
            if pltf_match:
                candidate = pltf_match.group(1).strip()
                if not re.search(r'MUNICIPAL COURT|FRANKLIN COUNTY', candidate, re.IGNORECASE):
                    plaintiff = candidate
        
        # Method 4: Use signature tag content (for handwritten/signed names)
        if not plaintiff and sig_names:
            for sig in sig_names:
                if len(sig.strip()) <= 1:  # v3.9: Skip single-char signatures
                    continue
                sig_pos = text.find(f'<signature>{sig}</signature>')
                after_sig = text[sig_pos:sig_pos+200].lower()
                # v3.9.1: Check which label appears first (not mutual exclusion)
                p_idx = after_sig.find('plaintiff')
                d_idx = after_sig.find('defendant')
                if p_idx != -1 and (d_idx == -1 or p_idx < d_idx):
                    plaintiff = sig.strip()
                    break
        
        # === DEFENDANT EXTRACTION (text-based methods) ===
        
        # Method 1: Name on line after "v." (standard format)
        if not defendant:
            def_match = re.search(
                r'(?:v\.|Vs|versus|vs)\s*\.?\s*\n\s*(?:Name:\s*)?([^\n]+)',
                text_no_sigs, re.MULTILINE | re.IGNORECASE
            )
            if def_match:
                candidate = def_match.group(1).strip()
                candidate = re.sub(r'Address:.*$', '', candidate, flags=re.IGNORECASE)
                candidate = re.sub(r'Name:\s*', '', candidate)
                candidate = re.sub(r'[,\s]*(?:Defendant|DEFENDANT).*$', '', candidate, flags=re.IGNORECASE)
                candidate = re.sub(r'\s+', ' ', candidate).strip()
                if candidate and len(candidate) > 1:
                    defendant = candidate
        
        # Method 2: "v. Name" on same line
        if not defendant:
            inline_def = re.search(
                r'(?:v\.|vs\.|versus)\s+([A-Z][A-Za-z\s]+?)(?:\s*,|\s*\n)',
                text_no_sigs, re.IGNORECASE
            )
            if inline_def:
                candidate = inline_def.group(1).strip()
                candidate = re.sub(r'[,\s]*(?:Defendant|DEFENDANT).*$', '', candidate, flags=re.IGNORECASE)
                if candidate and len(candidate) > 1:
                    defendant = candidate
        
        # Method 3: "DEFENDANT(S)" format  
        if not defendant:
            alt_def = re.search(
                r'(?:v\.|vs\.)\s*\n?\s*([A-Z][A-Za-z\s.,]+?)\s*(?:,\s*)?(?:DEFENDANT\(S\)|Defendant\(s\))',
                text_no_sigs, re.IGNORECASE | re.DOTALL
            )
            if alt_def:
                defendant = alt_def.group(1).strip()
        
        # Method 4: Use signature tag content
        # v3.9: Skip single-character signatures and use tighter window
        if not defendant and sig_names:
            for sig in sig_names:
                if len(sig.strip()) <= 1:  # v3.9: Skip single-char like "J", "X", "Q"
                    continue
                sig_pos = text.find(f'<signature>{sig}</signature>')
                after_sig = text[sig_pos:sig_pos+200].lower()
                # v3.9.1: Check which label appears first (not mutual exclusion)
                d_idx = after_sig.find('defendant')
                p_idx = after_sig.find('plaintiff')
                if d_idx != -1 and (p_idx == -1 or d_idx < p_idx):
                    defendant = sig.strip()
                    break
        
        # === FINAL CLEANUP ===
        if plaintiff:
            plaintiff = re.sub(r'<[^>]+>', '', plaintiff).strip()
            plaintiff = re.sub(r'\s*[:/]\s*$', '', plaintiff)
            plaintiff = re.sub(r',\s*$', '', plaintiff)
            if len(plaintiff) > 100 or len(plaintiff) <= 1 or plaintiff.lower().startswith('defendant') or not plaintiff:  # v3.9: reject single-char
                plaintiff = None
        
        if defendant:
            defendant = re.sub(r'<[^>]+>', '', defendant).strip()
            defendant = re.sub(r'\s*[,.]$', '', defendant)
            defendant = re.sub(r'^\(.*?\)\s*', '', defendant)  # Remove leading parens
            # v3.9: reject single-char names and the label word "Defendant(s)" itself
            if (len(defendant) > 100 or len(defendant) <= 1 or not defendant
                    or re.match(r'^Defendants?$', defendant, re.IGNORECASE)):
                defendant = None
        
        return plaintiff, defendant
    
    @staticmethod
    def extract_payment_schedule(text: str) -> Tuple[List[PaymentScheduleItem], Optional[str]]:
        """
        ENHANCED v3.9: Extract payment schedule with flexible multi-column table support.
        
        v3.9 additions:
        - Pattern 2f: "will pay $AMT no later than DATE" and
          "MONTH rent will be paid no later than DATE"
        - Improved 5-column table: strips leading punctuation/bullets from cells
          before checking for $ (handles "; • $ 975.00")
        
        Handles:
        - 2-col: [amount, date]
        - 3-col: [amount, date, extra] or [number, amount, date]
        - 4-col: [number, amount_description, date, time]
        - 5-col: paired amount+date columns with bullet separators
        - Descriptive amounts: "$Feb rent + late fee", "$March rent"
        - OCR artifacts: "$Apri1 rent" -> April rent
        """
        payments = []
        payment_num = 1
        total_sum = 0.0
        
        def parse_time(time_str: Optional[str]) -> Optional[str]:
            if not time_str:
                return None
            return time_str.strip()
        
        def parse_payment_description(text: str) -> dict:
            result = {
                'payment_type': 'fixed_amount',
                'month_rent': None,
                'extra_text': None
            }
            if not text:
                return result
            
            # Check for month rent pattern (full or abbreviated month names)
            month_pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+rent'
            month_match = re.search(month_pattern, text, re.IGNORECASE)
            
            if month_match:
                result['payment_type'] = 'monthly_rent'
                result['month_rent'] = month_match.group(1).title()
            else:
                # Also check for OCR-corrupted month names like "Apri1"
                ocr_month_pattern = r'(Jan\w*|Feb\w*|Mar\w*|Apr\w*|May|Jun\w*|Jul\w*|Aug\w*|Sep\w*|Oct\w*|Nov\w*|Dec\w*)\s+rent'
                ocr_match = re.search(ocr_month_pattern, text, re.IGNORECASE)
                if ocr_match:
                    result['payment_type'] = 'monthly_rent'
                    month_name = re.sub(r'[0-9]', '', ocr_match.group(1))  # Remove OCR digit artifacts
                    result['month_rent'] = month_name.title()
            
            extras = []
            if 'water' in text.lower():
                extras.append('water')
            if 'late fee' in text.lower() or 'late charges' in text.lower():
                extras.append('late fees')
            if 'utility' in text.lower() or 'utilities' in text.lower():
                extras.append('utilities')
            if 'third party' in text.lower():
                extras.append('third party assistance')
            
            if extras:
                result['extra_text'] = ', '.join(extras)
            
            return result
        
        # ============================================================
        # PATTERN 1: FLEXIBLE Table format (handles 2, 3, 4, or 5 columns)
        # Identifies amount/date/time cells by CONTENT, not position
        # v3.9: Strips leading punctuation/bullets from cells before $ detection
        # ============================================================
        table_rows = re.findall(r'<tr>(.*?)</tr>', text, re.DOTALL)
        
        for row in table_rows:
            cells = re.findall(r'<td>(.*?)</td>', row, re.DOTALL)
            cells = [c.strip() for c in cells]
            
            if len(cells) < 2:
                continue
            
            # Skip blank/template rows (all underlines, no real data)
            # Also count if we have a date-like cell
            has_date_cell = any(re.search(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', c) for c in cells)
            if not has_date_cell:
                continue
            
            # ====== Identify cells by content type ======
            # v3.9: Strip leading punctuation/bullets before checking for $
            # This handles cells like "; • $ 975.00" from 5-column tables
            amount_cells = []
            date_cells = []
            time_cell_idx = None
            
            for idx, cell in enumerate(cells):
                # v3.9: Clean cell for detection (strip bullets, semicolons, unicode chars)
                # v3.9.1: Also strip leading row numbers like "1." "2." etc.
                cleaned_cell = re.sub(r'^(?:\d+\.\s*)?[;\s•·\-–—\u2022\u00e2\u0080\u00a2]*', '', cell).strip()
                
                # Dollar amount (numeric like $2538.52 or descriptive like $Feb rent)
                # Also handle split $ cells: ['$', '300', ...] where $ is alone
                if cleaned_cell.startswith('$') and len(cleaned_cell) > 1:
                    amount_cells.append(idx)
                elif cleaned_cell == '$' and idx + 1 < len(cells):
                    amount_cells.append(idx)
                # Date pattern (slash, dash, or dot separated)
                elif re.search(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', cell):
                    date_cells.append(idx)
                # Time pattern (e.g., "by 5:00 PM")
                elif re.search(r'(?:by\s+)?\d{1,2}:\d{2}\s*(?:AM|PM)?', cell, re.IGNORECASE):
                    time_cell_idx = idx
            
            # Check for combined "$amount by date" in any cell
            if not amount_cells or not date_cells:
                for idx, cell in enumerate(cells):
                    combined_match = re.search(
                        r'\$\s*(\d+(?:\.\d{2})?)\s+(?:by|on or before)?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
                        cell, re.IGNORECASE
                    )
                    if combined_match:
                        amount_str = combined_match.group(1)
                        date_str = combined_match.group(2).replace('-', '/').replace('.', '/')
                        time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM)?)', cell, re.IGNORECASE)
                        time_str = time_match.group(1) if time_match else None
                        payment_info = parse_payment_description(cell)
                        
                        try:
                            total_sum += float(amount_str)
                        except:
                            pass
                        
                        payments.append(PaymentScheduleItem(
                            payment_number=payment_num,
                            due_date=normalize_date(date_str),
                            payment_type=payment_info['payment_type'],
                            amount=amount_str,
                            month_rent=payment_info['month_rent'],
                            time=time_str,
                            extra_text=payment_info['extra_text'],
                            raw_text=f"${amount_str} by {date_str}"
                        ))
                        payment_num += 1
                        break
                    # Also check for combined with "rent" like "$700 rent on or before 7/17/24 by 5:00 PM"
                    combined_rent_match = re.search(
                        r'\$\s*(\d+(?:\.\d{2})?)\s+(?:rent\s+)?(?:on\s+or\s+before|by)\s+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
                        cell, re.IGNORECASE
                    )
                    if combined_rent_match:
                        amount_str = combined_rent_match.group(1)
                        date_str = combined_rent_match.group(2).replace('-', '/').replace('.', '/')
                        time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM)?)', cell, re.IGNORECASE)
                        time_str = time_match.group(1) if time_match else None
                        payment_info = parse_payment_description(cell)
                        
                        try:
                            total_sum += float(amount_str)
                        except:
                            pass
                        
                        payments.append(PaymentScheduleItem(
                            payment_number=payment_num,
                            due_date=normalize_date(date_str),
                            payment_type=payment_info['payment_type'],
                            amount=amount_str,
                            month_rent=payment_info['month_rent'],
                            time=time_str,
                            extra_text=payment_info['extra_text'],
                            raw_text=f"${amount_str} by {date_str}"
                        ))
                        payment_num += 1
                        break
                continue
            
            # Pair up amount cells with their nearest date cell
            # In 4-column tables: col 0+1 = pair 1, col 2+3 = pair 2
            pairs = []
            used_dates = set()
            for a_idx in amount_cells:
                # Find nearest date cell AFTER this amount cell
                best_d = None
                for d_idx in date_cells:
                    if d_idx > a_idx and d_idx not in used_dates:
                        best_d = d_idx
                        break
                if best_d is not None:
                    pairs.append((a_idx, best_d))
                    used_dates.add(best_d)
            
            for amount_cell_idx, date_cell_idx in pairs:
                # v3.9: Clean the amount cell (strip leading bullets/punctuation)
                # v3.9.1: Also strip leading row numbers like "1." "2." etc.
                raw_amount_cell = cells[amount_cell_idx]
                cleaned_amount_cell = re.sub(r'^(?:\d+\.\s*)?[;\s•·\-–—\u2022\u00e2\u0080\u00a2]*', '', raw_amount_cell).strip()
                
                # Handle split $ cell: if amount_cell is just '$', merge with next cell
                if cleaned_amount_cell == '$' and amount_cell_idx + 1 < len(cells):
                    next_idx = amount_cell_idx + 1
                    if next_idx == date_cell_idx:
                        continue  # $ is alone and next cell is the date → blank amount
                    amount_cell = '$' + cells[next_idx]
                else:
                    amount_cell = cleaned_amount_cell
                date_cell = cells[date_cell_idx]
                
                date_match = re.search(r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})', date_cell)
                if not date_match:
                    continue
                date_str = date_match.group(1).replace('-', '/').replace('.', '/')
                
                # Extract time
                time_str = None
                if time_cell_idx is not None:
                    time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM)?)', cells[time_cell_idx], re.IGNORECASE)
                    if time_match:
                        time_str = time_match.group(1)
                if not time_str:
                    time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM)?)', date_cell, re.IGNORECASE)
                    if time_match:
                        time_str = time_match.group(1)
                
                # Parse the amount cell
                numeric_amount_match = re.search(r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', amount_cell)
                
                # Skip if amount cell is blank/template (just $ or $____)
                amount_content = re.sub(r'[\$\s_.]', '', amount_cell)
                if not amount_content and not numeric_amount_match:
                    continue
                
                if numeric_amount_match:
                    amount = numeric_amount_match.group(1).replace(',', '')
                    payment_info = parse_payment_description(amount_cell)
                    
                    try:
                        total_sum += float(amount)
                    except:
                        pass
                    
                    payments.append(PaymentScheduleItem(
                        payment_number=payment_num,
                        due_date=normalize_date(date_str),
                        payment_type=payment_info['payment_type'],
                        amount=amount,
                        month_rent=payment_info['month_rent'],
                        time=time_str,
                        extra_text=payment_info['extra_text'] if payment_info['extra_text'] else None,
                        raw_text=f"${amount} by {date_str}"
                    ))
                    payment_num += 1
                else:
                    # Descriptive amount like "$Feb rent + late fee"
                    desc = amount_cell.lstrip('$').strip()
                    if desc:
                        payment_info = parse_payment_description(desc)
                        
                        payments.append(PaymentScheduleItem(
                            payment_number=payment_num,
                            due_date=normalize_date(date_str),
                            payment_type=payment_info['payment_type'],
                            amount=None,
                            month_rent=payment_info['month_rent'],
                            time=time_str,
                            extra_text=desc,
                            raw_text=f"${desc} by {date_str}"
                        ))
                        payment_num += 1
        
        # ============================================================
        # PATTERN 2: Non-table payment formats (bullet, inline, prose)
        # v3.9: Added Pattern 2f for "will pay" and "no later than" variants
        # v3.8: Unified flexible patterns supporting:
        #   - Slash and dash date separators (1/15/24 or 1-15-24)
        #   - 2 and 4 digit years
        #   - "on or before", "by", "due", "due to Plaintiff by"
        #   - "to be paid by", "to be paid no later than"
        #   - Text dates (January 15, 2024 or Jan 15, 2024)
        #   - Multi-payment lines (a. $4000 by 11-25-24 d. $5000 by 12-17-24)
        #   - "agrees to pay $AMT by TIME, DATE" (inline sentence)
        # ============================================================
        
        # Date sub-patterns for reuse
        _NUMERIC_DATE = r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}'
        _TEXT_DATE_FULL = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}'
        _TEXT_DATE_ABBR = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}'
        _ANY_DATE = rf'(?:{_NUMERIC_DATE}|{_TEXT_DATE_FULL}|{_TEXT_DATE_ABBR})'
        _TIME_PATTERN = r'(\d{1,2}:\d{2}\s*(?:AM|PM)?)'
        
        if not payments:
            # --- Pattern 2a: "$AMT on or before|by DATE" (standard bullet) ---
            # Handles slash & dash dates, 2 & 4 digit years, with optional time
            bullet_pattern = (
                r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s+'
                r'(?:on\s+or\s+before|by|due\s+(?:to\s+\w+\s+)?by)\s+'
                rf'({_NUMERIC_DATE})'
                r'(?:\s+(?:by\s+)?(\d{1,2}:\d{2}\s*(?:AM|PM)))?'
            )
            for match in re.finditer(bullet_pattern, text, re.IGNORECASE):
                amount = match.group(1).replace(',', '')
                date = match.group(2).replace('-', '/').replace('.', '/')
                time = match.group(3) if match.group(3) else None
                
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end]
                payment_info = parse_payment_description(context)
                
                try:
                    total_sum += float(amount)
                except:
                    pass
                
                payments.append(PaymentScheduleItem(
                    payment_number=payment_num,
                    due_date=normalize_date(date),
                    payment_type=payment_info['payment_type'],
                    amount=amount,
                    month_rent=payment_info['month_rent'],
                    time=parse_time(time),
                    extra_text=payment_info['extra_text'],
                    raw_text=match.group(0)
                ))
                payment_num += 1
        
        if not payments:
            # --- Pattern 2b: "$AMT on or before|by|due TEXT_DATE" ---
            # e.g., "$6560 due July 25, 2024" or "$2000 by December 15, 2024"
            text_date_pattern = (
                r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s+'
                r'(?:on\s+or\s+before|by|due\s+(?:to\s+\w+\s+)?by|due)\s+'
                rf'({_TEXT_DATE_FULL}|{_TEXT_DATE_ABBR})'
            )
            for match in re.finditer(text_date_pattern, text, re.IGNORECASE):
                amount = match.group(1).replace(',', '')
                date = match.group(2)
                
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end]
                payment_info = parse_payment_description(context)
                
                try:
                    total_sum += float(amount)
                except:
                    pass
                
                payments.append(PaymentScheduleItem(
                    payment_number=payment_num,
                    due_date=normalize_date(date),
                    payment_type=payment_info['payment_type'],
                    amount=amount,
                    month_rent=payment_info['month_rent'],
                    time=None,
                    extra_text=payment_info['extra_text'],
                    raw_text=match.group(0)
                ))
                payment_num += 1
        
        if not payments:
            # --- Pattern 2c: "$AMT to be paid (by|no later than) DATE" ---
            # Prose format common in attorney-drafted entries
            prose_pattern = (
                r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
                r'([^$\n]*?)'  # Intervening text - don't cross $ signs or newlines
                r'(?:to\s+be\s+paid|due)\s+'
                r'(?:by|no\s+later\s+than)\s+'
                rf'({_TEXT_DATE_FULL}|{_TEXT_DATE_ABBR}|{_NUMERIC_DATE})'
            )
            for match in re.finditer(prose_pattern, text, re.IGNORECASE):
                amount = match.group(1).replace(',', '')
                intervening = match.group(2).lower()
                date = match.group(3).replace('-', '/').replace('.', '/')
                
                # Avoid matching summary/description amounts
                if 'represents' in intervening or 'in the amount of' in intervening:
                    continue
                
                context_start = max(0, match.start() - 80)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end]
                payment_info = parse_payment_description(context)
                
                try:
                    total_sum += float(amount)
                except:
                    pass
                
                payments.append(PaymentScheduleItem(
                    payment_number=payment_num,
                    due_date=normalize_date(date),
                    payment_type=payment_info['payment_type'],
                    amount=amount,
                    month_rent=payment_info['month_rent'],
                    time=None,
                    extra_text=payment_info['extra_text'],
                    raw_text=match.group(0)
                ))
                payment_num += 1
        
        if not payments:
            # --- Pattern 2d: "MONTH rent of $AMT to be paid by DATE" ---
            rent_prose_pattern = (
                r'(\w+)\s+rent\s+of\s+\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
                r'[^.]*?'
                r'(?:to\s+be\s+paid|due)\s+'
                r'(?:by|no\s+later\s+than)\s+'
                rf'({_TEXT_DATE_FULL}|{_TEXT_DATE_ABBR}|{_NUMERIC_DATE})'
            )
            for match in re.finditer(rent_prose_pattern, text, re.IGNORECASE):
                month_name = match.group(1).title()
                amount = match.group(2).replace(',', '')
                date = match.group(3).replace('-', '/').replace('.', '/')
                
                try:
                    total_sum += float(amount)
                except:
                    pass
                
                payments.append(PaymentScheduleItem(
                    payment_number=payment_num,
                    due_date=normalize_date(date),
                    payment_type='monthly_rent',
                    amount=amount,
                    month_rent=month_name,
                    time=None,
                    extra_text=None,
                    raw_text=match.group(0)
                ))
                payment_num += 1
        
        if not payments:
            # --- Pattern 2e: "agrees to pay $AMT by TIME, DATE" (inline sentence) ---
            inline_pay_pattern = (
                r'(?:agrees?\s+to\s+pay|shall\s+pay)[^$]*?'
                r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s+'
                r'by\s+'
                r'(?:(\d{1,2}:\d{2}\s*(?:AM|PM)),?\s*)?'
                rf'({_TEXT_DATE_FULL}|{_TEXT_DATE_ABBR}|{_NUMERIC_DATE})'
            )
            for match in re.finditer(inline_pay_pattern, text, re.IGNORECASE):
                amount = match.group(1).replace(',', '')
                time = match.group(2) if match.group(2) else None
                date = match.group(3).replace('-', '/').replace('.', '/')
                # Clean ordinal suffixes from date
                date = re.sub(r'(\d{1,2})(?:st|nd|rd|th)', r'\1', date)
                
                try:
                    total_sum += float(amount)
                except:
                    pass
                
                payments.append(PaymentScheduleItem(
                    payment_number=payment_num,
                    due_date=normalize_date(date),
                    payment_type='fixed_amount',
                    amount=amount,
                    month_rent=None,
                    time=parse_time(time),
                    extra_text=None,
                    raw_text=match.group(0)
                ))
                payment_num += 1
        
        if not payments:
            # --- Pattern 2f (v3.9): "will pay $AMT no later than DATE" ---
            # Also handles: "$AMT no later than DATE" (bare),
            #   "will pay $AMT no later than DAYNAME DATE"
            # And: "MONTH rent will be paid no later than DATE"
            # e.g., "Defendant will pay $6,371.64 no later than Monday June 24, 2024."
            # e.g., "July rent will be paid no later than July 5, 2024."
            
            # Sub-pattern A: "will/shall pay $AMT no later than [DAYNAME] DATE"
            will_pay_pattern = (
                r'(?:will\s+pay|shall\s+pay|agrees?\s+to\s+pay)[^$\n]*?'
                r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s+'
                r'(?:no\s+later\s+than|by|on\s+or\s+before)\s+'
                r'(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+)?'
                rf'({_TEXT_DATE_FULL}|{_TEXT_DATE_ABBR}|{_NUMERIC_DATE})'
            )
            for match in re.finditer(will_pay_pattern, text, re.IGNORECASE):
                amount = match.group(1).replace(',', '')
                date = match.group(2).replace('-', '/').replace('.', '/')
                
                context_start = max(0, match.start() - 80)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end]
                payment_info = parse_payment_description(context)
                
                try:
                    total_sum += float(amount)
                except:
                    pass
                
                payments.append(PaymentScheduleItem(
                    payment_number=payment_num,
                    due_date=normalize_date(date),
                    payment_type=payment_info['payment_type'],
                    amount=amount,
                    month_rent=payment_info['month_rent'],
                    time=None,
                    extra_text=payment_info['extra_text'],
                    raw_text=match.group(0)
                ))
                payment_num += 1
            
            # Sub-pattern B: "MONTH rent will be paid no later than DATE"
            # (no dollar amount — just a month rent reference)
            rent_no_later_pattern = (
                r'(January|February|March|April|May|June|July|August|September|October|November|December'
                r'|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+'
                r'rent\s+(?:will\s+be|shall\s+be|to\s+be|is\s+to\s+be)\s+paid\s+'
                r'(?:no\s+later\s+than|by|on\s+or\s+before)\s+'
                r'(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+)?'
                rf'({_TEXT_DATE_FULL}|{_TEXT_DATE_ABBR}|{_NUMERIC_DATE})'
            )
            for match in re.finditer(rent_no_later_pattern, text, re.IGNORECASE):
                month_name = match.group(1).title()
                date = match.group(2).replace('-', '/').replace('.', '/')
                
                payments.append(PaymentScheduleItem(
                    payment_number=payment_num,
                    due_date=normalize_date(date),
                    payment_type='monthly_rent',
                    amount=None,
                    month_rent=month_name,
                    time=None,
                    extra_text=None,
                    raw_text=match.group(0)
                ))
                payment_num += 1
        
        # ============================================================
        # PATTERN 3: Description-based format (rent + extras)
        # ============================================================
        if not payments:
            desc_pattern = r'\$([A-Za-z]+\s+rent[^$\n]*?)\s+on or before\s+(\d{1,2}/\d{1,2}/\d{4})(?:\s+(?:by\s+)?(\d{1,2}:\d{2}\s*(?:AM|PM)))?'
            for match in re.finditer(desc_pattern, text, re.IGNORECASE):
                description = match.group(1)
                date = match.group(2)
                time = match.group(3) if match.group(3) else None
                
                payment_info = parse_payment_description(description)
                
                payments.append(PaymentScheduleItem(
                    payment_number=payment_num,
                    due_date=normalize_date(date),
                    payment_type='monthly_rent',
                    amount=None,
                    month_rent=payment_info['month_rent'],
                    time=parse_time(time),
                    extra_text=description,
                    raw_text=match.group(0)
                ))
                payment_num += 1
        
        total_sum_str = f"{total_sum:.2f}" if total_sum > 0 else None
        return payments, total_sum_str
    
    @staticmethod
    def extract_vacate_date(text: str) -> Optional[str]:
        """
        Extract mandatory vacate date.
        
        CRITICAL: Only matches ACTUAL UNCONDITIONAL vacate obligations.
        
        DOES NOT match:
        - "Notice to Vacate" (procedural acknowledgment)
        - "agrees to vacate" in BREACH clauses (conditional consequence)
        
        v3.9: Added "premises?" (singular) support and optional adverb
              (e.g., "voluntarily vacate the premise")
        v3.8: Added pattern for abbreviated "D Shall vacate by DATE"
        v3.7: Handles ordinal dates (26th), markdown bold (**), and
              long clauses between "vacate premises" and date.
        """
        # Pre-clean text: strip markdown bold markers and ordinal suffixes
        cleaned = re.sub(r'\*\*', '', text)  # Remove ** bold markers
        cleaned = re.sub(r'(\d{1,2})(?:st|nd|rd|th)', r'\1', cleaned)  # "26th" → "26"
        
        # v3.9: Date sub-pattern for vacate — accepts slash, dash, and dot separators
        _VACATE_DATE = r'[A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}'
        
        patterns = [
            # Pattern 1: "agrees to vacate [voluntarily] the premises/premise[, ...stuff...], on or before DATE"
            # v3.9: "premises?" allows singular, "\w+\s+" allows optional adverb like "voluntarily"
            # Allow up to 200 chars of intervening text for long clauses
            rf'(?:shall|must|agree(?:s)?\s+to|required\s+to)\s+(?:\w+\s+)?vacate\s+(?:the\s+)?(?:premises?|property|rental\s+unit)[^.]*?(?:on\s+or\s+before|by|before|on)\s+({_VACATE_DATE})',
            # Pattern 2: "Defendant shall vacate ... by DATE"
            # v3.9: Added optional adverb and singular "premise"
            rf'Defendant(?:\(s\))?\s+(?:shall|must|agrees?\s+to)\s+(?:\w+\s+)?vacate[^.]*?(?:by|before|on\s+or\s+before|on)\s+({_VACATE_DATE})',
            # Pattern 3: "vacate the premises/premise ... by DATE"
            # v3.9: singular support
            rf'vacate\s+(?:the\s+)?premises?[^.]*?(?:on\s+or\s+before|by|before|on)\s+({_VACATE_DATE})',
            # Pattern 4 (v3.8): Abbreviated "D" for Defendant or bare "Shall vacate by DATE"
            rf'(?:^|\n)\s*\*?\s*(?:D|Def\.?|Dft\.?|Defendant(?:\(s\))?)\s+[Ss]hall\s+vacate\s+(?:by|on\s+or\s+before|before)\s+({_VACATE_DATE})',
            # Pattern 5 (v3.9): "vacate the premise and turn in keys on or before DATE"
            # Handles the exact phrasing where "premise" is followed by "and turn in keys"
            # rather than directly by "on or before"
            rf'vacate\s+(?:the\s+)?premises?\s+and\s+[^.]*?(?:on\s+or\s+before|by|before)\s+({_VACATE_DATE})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE | re.MULTILINE)
            if match:
                # Check context BEFORE the match for exclusion phrases
                context_start = max(0, match.start() - 150)
                context = cleaned[context_start:match.start()].lower()
                
                # Skip if procedural acknowledgment of notice served
                if any(phrase in context for phrase in [
                    'acknowledges receipt', 'receipt of',
                    'notice to vacate served', 'notice to leave served'
                ]):
                    continue
                
                # Skip if inside a breach/default consequence clause
                if any(phrase in context for phrase in [
                    'breach', 'default', 'fail to comply', 'failure to comply',
                    'if any of the terms', 'if the terms',
                    'not object to', 'granted judgment',
                    'set-out', 'set out'
                ]):
                    continue
                    
                return normalize_date(match.group(1).strip())
        
        return None
    
    @staticmethod
    def detect_breach_clause_vacate(text: str) -> bool:
        """
        v3.8: Explicitly detect whether the document's vacate language appears
        ONLY inside a breach/default consequence clause.
        
        Returns True if the document mentions vacating, but ONLY in breach context
        (e.g., "if terms are breached... agrees to vacate immediately").
        
        v3.9: Updated to match "premises?" (singular) and optional adverb
        """
        text_lower = text.lower()
        
        # Find ALL occurrences of vacate-related language
        # v3.9: Added "premises?" and optional adverb patterns
        vacate_mentions = list(re.finditer(
            r'(?:shall|must|agree(?:s)?\s+to)\s+(?:\w+\s+)?vacate|vacate\s+(?:the\s+)?(?:premises?|property|immediately)',
            text_lower
        ))
        
        if not vacate_mentions:
            return False
        
        # Check each vacate mention to see if it's in a breach context
        breach_indicators = [
            'breach', 'default', 'fail to comply', 'failure to comply',
            'if any of the terms', 'if the terms',
            'not object to', 'granted judgment',
            'set-out', 'set out'
        ]
        
        notice_indicators = [
            'acknowledges receipt', 'receipt of',
            'notice to vacate served', 'notice to leave served'
        ]
        
        non_breach_vacate_found = False
        
        for match in vacate_mentions:
            context_start = max(0, match.start() - 200)
            context_before = text_lower[context_start:match.start()]
            
            # Is this vacate in a breach clause?
            in_breach = any(phrase in context_before for phrase in breach_indicators)
            
            # Is this vacate just acknowledging a notice?
            in_notice = any(phrase in context_before for phrase in notice_indicators)
            
            if not in_breach and not in_notice:
                # Found a vacate mention that is NOT in breach context and NOT a notice
                non_breach_vacate_found = True
                break
        
        # Return True if ALL vacate mentions are in breach/notice context
        return not non_breach_vacate_found
    
    @staticmethod
    def extract_outcome_type(text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Determine outcome type with improved logic.
        
        v3.5 FIXES:
        - Uses extract_vacate_date() which excludes breach-conditional vacate
        - Better payment detection for descriptive amounts ($Feb rent, etc.)
        - Detects "agree to pay ... in the following manner" as payment signal
        
        Returns 3 main categories:
        1. Vacate Only - Agree to vacate before a certain date
        2. Pay and Stay - Agree to pay (and stay in the home)
        3. Pay and Vacate - Agree to pay and then vacate before a certain date
        """
        text_lower = text.lower()
        
        # Detect payments - multiple methods
        payment_match = re.search(r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', text)
        has_payment = payment_match is not None
        
        # Also check for descriptive payment patterns like "$Feb rent"
        if not has_payment:
            has_payment = bool(re.search(
                r'\$\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+rent',
                text, re.IGNORECASE
            ))
        
        # Also check for payment schedule language (multiple phrasings)
        if not has_payment:
            has_payment = bool(re.search(
                r'agree\s+to\s+pay.*?(?:following|schedule|manner)|'
                r'shall\s+make\s+the\s+following\s+payments|'
                r'due\s+to\s+plaintiff\s+by|'
                r'(?:agrees?\s+to\s+pay|shall\s+pay)\s+(?:plaintiff\s+)?\$',
                text_lower
            ))
        
        # v3.9: Also check for "will pay $" pattern
        if not has_payment:
            has_payment = bool(re.search(
                r'will\s+pay\s+\$|'
                r'\w+\s+rent\s+(?:will|shall)\s+be\s+paid',
                text_lower
            ))
        
        # Also check for OCR-mangled dollar amounts ($ #1153, $ #2965)
        if not has_payment:
            has_payment = bool(re.search(r'\$\s*#\d+', text))
        
        # Use the precise extract_vacate_date() which excludes false positives
        vacate_date = RegexExtractor.extract_vacate_date(text)
        has_vacate = vacate_date is not None
        
        has_dismissal = bool(re.search(r'dismiss.*?(?:if|upon).*?(?:compliance|terms?|satisfied|met)', text_lower, re.DOTALL))
        has_payment_plan = bool(re.search(r'payment.*?(?:schedule|plan)|pay.*?follows|installments|following\s+manner', text_lower))
        has_immediate_judgment = bool(re.search(r'immediate.*?judgment|move.*?directly.*?judgment|breach.*?judgment', text_lower, re.DOTALL))
        
        outcome_type = None
        outcome_details = None
        
        if has_payment and has_vacate:
            outcome_type = "Pay and Vacate"
            details = []
            if has_payment_plan:
                details.append("Payment Plan")
            if vacate_date:
                details.append(f"Vacate by {vacate_date}")
            if has_dismissal:
                details.append("Dismiss if Terms Met")
            if has_immediate_judgment:
                details.append("Immediate Judgment on Breach")
            outcome_details = " - ".join(details) if details else "Payment + Vacate Agreement"
            
        elif has_vacate and not has_payment:
            outcome_type = "Vacate Only"
            details = []
            if vacate_date:
                details.append(f"By {vacate_date}")
            if has_dismissal:
                details.append("Dismiss if Vacated")
            if has_immediate_judgment:
                details.append("Immediate Judgment on Breach")
            outcome_details = " - ".join(details) if details else "Vacate Agreement"
            
        elif has_payment and not has_vacate:
            outcome_type = "Pay and Stay"
            details = []
            if has_payment_plan:
                details.append("Payment Plan")
            else:
                details.append("Lump Sum Payment")
            if has_dismissal:
                details.append("Dismiss if Paid")
            if has_immediate_judgment:
                details.append("Immediate Judgment on Breach")
            outcome_details = " - ".join(details) if details else "Payment Agreement"
            
        else:
            if 'agreed judgment entry' in text_lower:
                outcome_type = "Agreed Judgment"
                outcome_details = "Terms Require Review"
            elif 'agreed entry' in text_lower:
                outcome_type = "Agreed Entry"
                outcome_details = "Terms Require Review"
            elif 'settlement' in text_lower:
                outcome_type = "Settlement"
                outcome_details = "Terms Require Review"
            else:
                outcome_type = "Unknown"
                outcome_details = "Unable to Determine Terms"
        
        return outcome_type, outcome_details
    
    @staticmethod
    def check_third_party_acceptance(text: str) -> Optional[bool]:
        """Check if third-party assistance/payment is mentioned"""
        patterns = [
            r'third[\s-]?party\s+(?:assistance|payment)',
            r'accept\s+third[\s-]?party',
            r'third party assistance will be accepted',
            r'3rd party agency providing monetary assistance',
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return None
    
    @staticmethod
    def extract_assistance_deadline(text: str) -> Optional[str]:
        """Extract deadline for seeking assistance"""
        patterns = [
            r'(?:assistance|help|support)\s+(?:must be|shall be)\s+(?:received|obtained|secured)\s+by\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{4})',
            r'(?:by|before)\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{4})[^.]*?assistance',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return normalize_date(match.group(1).strip())
        return None
    
    @staticmethod
    def extract_additional_agreement_terms(text: str) -> Optional[str]:
        """Extract additional terms of the agreement"""
        terms = []
        
        term_patterns = {
            "dismiss_if_terms_met": r'dismiss.*?(?:if|upon).*?(?:compliance|terms?|satisfied|met)',
            "immediate_judgment_on_breach": r'(?:breach.*?judgment|immediate.*?judgment|move.*?directly.*?judgment)',
            "comply_with_lease": r'(?:strictly\s+)?comply\s+with\s+(?:all\s+)?lease',
            "pay_utilities": r'(?:timely\s+)?pay[^.]*?utilities',
            "no_contact": r'no\s+(?:intentional\s+)?contact',
            "return_keys": r'return[^.]*?keys',
            "broom_swept_clean": r'broom\s+(?:swept\s+)?clean',
            "remove_property": r'remove[^.]*?(?:personal\s+)?(?:property|possessions)',
            "vacate_premises": r'vacate\s+(?:the\s+)?premises?',  # v3.9: singular support
            "pest_control_compliance": r'compliance\s+with[^.]*?pest\s+control',
        }
        
        readable_terms = {
            "dismiss_if_terms_met": "Dismiss if terms met",
            "immediate_judgment_on_breach": "Immediate judgment on breach",
            "comply_with_lease": "Comply with lease",
            "pay_utilities": "Pay utilities",
            "no_contact": "No contact",
            "return_keys": "Return keys",
            "broom_swept_clean": "Broom swept clean",
            "remove_property": "Remove personal property",
            "vacate_premises": "Vacate premises",
            "pest_control_compliance": "Pest control compliance"
        }
        
        for term_name, pattern in term_patterns.items():
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                terms.append(readable_terms.get(term_name, term_name))
        
        return "; ".join(terms) if terms else None
    
    @staticmethod
    def extract_sealing_reference(text: str) -> Optional[str]:
        """Extract sealing/record references"""
        patterns = [
            r'sealed?\s+(?:from\s+)?online\s+(?:access|records?)',
            r'not object to.*?sealed?',
            r'case sealed',
            r'sealing.*?(?:records?|case)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        return None
    
    @staticmethod
    def extract_enforcement_period(text: str) -> Optional[str]:
        """Extract enforcement/validity period"""
        patterns = [
            r'(?:enforceable|in effect|valid)\s+(?:for\s+)?up to\s+(\d+\s+days)',
            r'(?:agreement|entry)\s+shall\s+remain\s+enforceable\s+for[^.]*?(\d+\s+days)',
            r'time for execution.*?extended.*?to\s+(\d+\s+days)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None


    @classmethod
    def extract_all(cls, text: str, filename: Optional[str] = None) -> ExtractedCourtInfo:
        """Run all regex extractors and return a populated Pydantic model."""
        plaintiff, defendant = cls.extract_parties(text)
        payments_raw, total_sum = cls.extract_payment_schedule(text)
        outcome_type, outcome_details = cls.extract_outcome_type(text)

        # Convert dataclass-style PaymentScheduleItems to Pydantic models
        payments = []
        for p in payments_raw:
            pt = None
            if hasattr(p, 'payment_type'):
                ptype = p.payment_type if isinstance(p, dict) else getattr(p, 'payment_type', None)
            else:
                ptype = p.get('payment_type') if isinstance(p, dict) else None
            
            if isinstance(p, dict):
                pt_str = p.get('payment_type', 'fixed_amount')
                pt = PaymentType.MONTHLY_RENT if pt_str == 'monthly_rent' else PaymentType.FIXED_AMOUNT
                payments.append(PaymentScheduleItem(
                    payment_number=p.get('payment_number', len(payments) + 1),
                    due_date=p.get('due_date'),
                    payment_type=pt,
                    amount=p.get('amount'),
                    month_rent=p.get('month_rent'),
                    time=p.get('time'),
                    extra_text=p.get('extra_text'),
                    raw_text=p.get('raw_text'),
                ))
            else:
                # It's the original dataclass PaymentScheduleItem
                ptype_val = getattr(p, 'payment_type', 'fixed_amount')
                pt = PaymentType.MONTHLY_RENT if ptype_val == 'monthly_rent' else PaymentType.FIXED_AMOUNT
                payments.append(PaymentScheduleItem(
                    payment_number=getattr(p, 'payment_number', len(payments) + 1),
                    due_date=getattr(p, 'due_date', None),
                    payment_type=pt,
                    amount=getattr(p, 'amount', None),
                    month_rent=getattr(p, 'month_rent', None),
                    time=getattr(p, 'time', None),
                    extra_text=getattr(p, 'extra_text', None),
                    raw_text=getattr(p, 'raw_text', None),
                ))

        # Map outcome_type string to enum
        outcome_enum = None
        if outcome_type:
            for member in OutcomeType:
                if member.value.lower() == outcome_type.lower():
                    outcome_enum = member
                    break

        return ExtractedCourtInfo(
            case_number=cls.extract_case_number(text, filename),
            agreement_signed_date=cls.extract_agreement_date(text, filename),
            plaintiff=plaintiff,
            defendant=defendant,
            payment_schedule=payments,
            total_payment_sum=total_sum,
            outcome_type=outcome_enum,
            outcome_details=outcome_details,
            mandatory_vacate_date=cls.extract_vacate_date(text),
            third_party_acceptance=cls.check_third_party_acceptance(text),
            assistance_deadline=cls.extract_assistance_deadline(text),
            additional_agreement_terms=cls.extract_additional_agreement_terms(text),
            sealing_reference_stipulation=cls.extract_sealing_reference(text),
            enforcement_period=cls.extract_enforcement_period(text),
            extraction_method="Regex",
            filename=filename,
            raw_text=text,
        )
