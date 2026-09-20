import csv
import re
import sys
from pathlib import Path

# Đảm bảo in UTF-8 trên Windows console
sys.stdout.reconfigure(encoding='utf-8')

D = Path('data/e_comercial')
REQ = ['doc_id', 'title', 'source_url', 'retrieved_at', 'document_version', 'audience']
mds = sorted(D.glob('*.md'))
sources_file = D / 'sources.csv'
rows = list(csv.DictReader(open(sources_file, encoding='utf-8')))
ids, auds = [], {}

print(f"{'FILE NAME':35} {'DOC_ID':30} {'AUDIENCE':10} {'STATUS'}")
print("-" * 85)

for p in mds:
    content = p.read_text(encoding='utf-8')
    parts = content.split('---')
    fm = {}
    if len(parts) >= 3:
        for line in parts[1].strip().split('\n'):
            line = line.strip()
            if ':' in line:
                k, v = line.split(':', 1)
                fm[k.strip()] = v.strip().strip('"\'')
    
    doc_id = fm.get('doc_id')
    ids.append(doc_id)
    aud = fm.get('audience', 'MISSING')
    auds[aud] = auds.get(aud, 0) + 1
    
    missing = [k for k in REQ if k not in fm]
    id_match = (doc_id == p.stem)
    
    if not missing and id_match:
        status = "OK"
    else:
        errs = []
        if missing:
            errs.append(f"Thieu {missing}")
        if not id_match:
            errs.append(f"doc_id ({doc_id}) != stem ({p.stem})")
        status = " | ".join(errs)
        
    print(f"{p.name:35} {str(doc_id):30} {str(aud):10} {status}")

print("-" * 85)
print('So file .md :', len(mds), '(can 5-10)')
csv_ids = sorted(r['doc_id'] for r in rows)
md_ids = sorted(filter(None, ids))
print('sources.csv :', 'KHOP 1-1' if csv_ids == md_ids else f'LECH (md: {len(md_ids)} vs csv: {len(csv_ids)})')
print('Audience    :', auds)
