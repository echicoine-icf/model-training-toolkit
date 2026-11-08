"""Project-neutral supervised training & holdout validation. No ELI hardcoding."""
import json
from pathlib import Path


def rows(path, *, allow_empty=False):
    result=[];ids=set();questions={}
    path=Path(path)
    with path.open(encoding='utf-8-sig') as f:
        for n,line in enumerate(f,1):
            if not line.strip():continue
            try:r=json.loads(line)
            except json.JSONDecodeError as exc:raise ValueError(f'{path}:{n}: invalid JSON: {exc}') from exc
            if not isinstance(r,dict) or not isinstance(r.get('question'),str) or not r['question'].strip() or 'expected' not in r:
                raise ValueError(f'{path}:{n}: need object with non-empty question and expected')
            if not isinstance(r['expected'],(str,dict,list)):
                raise ValueError(f'{path}:{n}: expected must be a string, JSON object or array')
            rid=str(r.get('id',n))
            if rid in ids:raise ValueError(f'{path}:{n}: duplicate ID {rid}')
            ids.add(rid)
            q=' '.join(r['question'].casefold().split())
            if q in questions:raise ValueError(f'{path}:{n}: repeated question also at {questions[q]}')
            questions[q]=n
            result.append(r)
    if not allow_empty and not result:raise ValueError(f'{path}: empty dataset')
    return result


def expected_text(obj):
    if isinstance(obj,str):return obj
    return json.dumps(obj,ensure_ascii=False,separators=(',',':'))


def validate(train,evalfile,*,allow_empty=False):
    a=rows(train,allow_empty=allow_empty);b=rows(evalfile,allow_empty=allow_empty)
    qa={' '.join(x['question'].casefold().split()) for x in a}
    qb={' '.join(x['question'].casefold().split()) for x in b}
    overlaps=qa&qb
    if overlaps:raise ValueError(f'Train/eval overlap: {len(overlaps)} questions; example: {next(iter(overlaps))!r}')
    # Eval questions must also have distinct IDs independent of train; IDs may repeat across splits.
    return {'train':len(a),'eval':len(b),'exact_question_overlap':0}
