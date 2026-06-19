import re
from typing import Dict, List, Any, Optional

class ABAParser:
    """
    Parseur de référence pour le corpus ABA.
    Consomme une ligne textuelle du corpus ABA et extrait les segments et métadonnées.
    """
    
    SEG_REGEX = {
        "A": re.compile(r"<SEG_A>\s*(.*?)\s*</SEG_A>"),
        "B": re.compile(r"<SEG_B>\s*(.*?)\s*</SEG_B>"),
        "A_PRIME": re.compile(r"<SEG_A_PRIME>\s*(.*?)\s*</SEG_A_PRIME>")
    }
    
    META_REGEX = re.compile(r"^(<(?:ADD|SUB|MUL|DIV)>)?(<(?:DX|LV)>)?(<(?:OUT|IN)>)?(<(?:ALPHA|OMEGA|A_PRIME)>)?\s*(.*?)(?:<EOL>|<EOS>|\s)*$")

    @classmethod
    def parse_line(cls, line: str) -> Optional[Dict[str, Any]]:
        line = line.strip()
        if not line:
            return None
            
        result = {"segments": {}}
        global_op = None
        
        for seg_name, regex in cls.SEG_REGEX.items():
            match = regex.search(line)
            if match:
                raw_content = match.group(1).strip()
                meta_match = cls.META_REGEX.match(raw_content)
                if meta_match:
                    op, chir, direc, pos, text = meta_match.groups()
                    if op and not global_op:
                        global_op = op.strip("<>")
                    
                    result["segments"][seg_name] = {
                        "text": text.strip(),
                        "chiralite": chir.strip("<>") if chir else None,
                        "direction": direc.strip("<>") if direc else None,
                        "position": pos.strip("<>") if pos else None
                    }
        
        result["op"] = global_op
        
        if len(result["segments"]) != 3:
            return None
            
        return result
