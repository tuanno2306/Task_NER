"""Đọc DDI XML, chia train/dev theo tài liệu và tạo BIO DataLoader."""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

TOKEN_PATTERN = r"\w+|[^\w\s]"
ENTITY_TYPES = ['drug', 'brand', 'group', 'drug_n']
LABELS = ['O'] + [f'{p}-{t}' for t in ENTITY_TYPES for p in ('B', 'I')]

def tokenize_words(text):
    matches = list(re.finditer(TOKEN_PATTERN, text))
    return [m.group() for m in matches], [(m.start(), m.end()) for m in matches]

def prepare_example(ex, label2id, policy="strict"):
    if policy not in {"strict", "project"}:
        raise ValueError(f"Unknown annotation policy: {policy}")
    words, offsets = tokenize_words(ex['text'])
    labels = [label2id['O']] * len(words)
    owners, issues = {}, []
    if not words:
        issues.append(dict(reason='empty_sentence', fatal=True))
    seen = set()
    # Deterministic longest-span-first projection, independent of XML entity order.
    entities = sorted(ex['entities'], key=lambda e: (-(e['end']-e['start']), e['start'], e['end'], e['type']))
    for ent in entities:
        if not 0 <= ent['start'] < ent['end'] <= len(ex['text']):
            issues.append(dict(reason='invalid_offset', entity=ent, fatal=True))
            continue
        key = (ent['start'], ent['end'], ent['type'])
        if key in seen:
            issues.append(dict(reason='duplicate_annotation', entity=ent, action='deduplicate'))
            continue
        seen.add(key)
        positions = [i for i, (s, e) in enumerate(offsets) if s < ent['end'] and e > ent['start']]
        if not positions:
            issues.append(dict(reason='no_token_for_entity', entity=ent, fatal=True))
            continue
        projected = dict(start=offsets[positions[0]][0], end=offsets[positions[-1]][1], type=ent['type'])
        if projected['start'] != ent['start'] or projected['end'] != ent['end']:
            issues.append(dict(reason='boundary_inside_token', entity=ent, projected=projected,
                               token_offsets=[offsets[i] for i in positions], action='expand_to_token_boundaries'))
            if policy == 'strict':
                continue
        shared = [i for i in positions if i in owners]
        if shared:
            issues.append(dict(reason='overlapping_entities', entity=ent,
                               conflicting_entities=[owners[i] for i in shared], action='drop_conflicting_span'))
            continue
        for j, i in enumerate(positions):
            labels[i] = label2id[f"{'B' if j == 0 else 'I'}-{ent['type']}"]
            owners[i] = ent
    return dict(text=ex['text'], words=words, offsets=offsets, labels=labels), issues
def audit_splits(splits, label2id, report_path, policy="strict"):
    from collections import Counter
    import warnings
    if policy not in {"strict", "project"}:
        raise ValueError(f"Unknown annotation policy: {policy}")
    prepared, report, summary = {}, {}, {}
    for name, examples in splits.items():
        prepared[name], report[name] = [], []
        for ex in examples:
            row, issues = prepare_example(ex, label2id, policy=policy)
            prepared[name].append(row)
            if issues:
                report[name].append(dict(id=ex.get('id'), document=ex.get('document'),
                                         text=ex['text'], issues=issues))
        counts = Counter(issue['reason'] for row in report[name] for issue in row['issues'])
        summary[name] = dict(sentences=len(examples), affected_sentences=len(report[name]),
                             reasons=dict(counts))
    report['_policy'] = policy
    report['_summary'] = summary
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    fatal = any(issue.get('fatal', False) for rows in report.values() if isinstance(rows, list)
                for row in rows for issue in row['issues'])
    if fatal or (policy == 'strict' and any(report[name] for name in splits)):
        raise ValueError(f'Annotation không hợp lệ; xem {report_path}')
    return prepared
def split_xml_files(folder, dev_size=0.1, seed=42):
    xml_files = sorted(Path(folder).rglob("*.xml"))

    train_files, dev_files = train_test_split(
        xml_files,
        test_size=dev_size,
        random_state=seed,
        shuffle=True,
    )

    return train_files, dev_files

def read_xml_file(path):
    examples = []
    for sent in ET.parse(path).getroot().iter("sentence"):
        text = sent.attrib.get("text", "")
        if not text.strip():
            if sent.findall("entity"):
                raise ValueError(f"Annotation trong câu rỗng: {path}")
            continue
        entities = []
        for ent in sent.findall("entity"):
            kind = ent.attrib["type"]
            if kind not in ENTITY_TYPES:
                raise ValueError(f"Loại entity không hỗ trợ: {kind}")
            for span in ent.attrib["charOffset"].split(";"):
                start, last = map(int, span.split("-"))
                end = last + 1
                if not 0 <= start < end <= len(text):
                    raise ValueError(f"Offset XML không hợp lệ: {path}, {span}")
                entities.append(dict(start=start, end=end, type=kind, text=text[start:end]))
        examples.append(dict(id=sent.attrib.get("id", ""), document=str(path),
                             text=text, entities=entities))
    return examples


def load_examples_from_files(xml_files):
    examples = []

    for xml_file in xml_files:
        examples.extend(read_xml_file(xml_file))

    return examples

class TokenDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return dict(words=row['words'], tags=torch.tensor(row['labels'], dtype=torch.long))
def collate_words(items):
    words = [item['words'] for item in items]
    lengths = torch.tensor([len(item['tags']) for item in items])
    if (lengths <= 0).any() or any(len(w) != len(item['tags']) for w, item in zip(words, items)):
        raise ValueError('Câu rỗng hoặc số token không khớp nhãn')
    tags = pad_sequence([item['tags'] for item in items], batch_first=True)
    mask = torch.arange(tags.size(1))[None, :] < lengths[:, None]
    return dict(words=words, tags=tags, mask=mask, lengths=lengths)



def make_loader(rows, batch_size=32, shuffle=False, seed=23022006):
    return DataLoader(TokenDataset(rows), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, collate_fn=collate_words,
                      generator=torch.Generator().manual_seed(seed))


def create_dataloaders(data_dir, output_dir, batch_size=32, seed=23022006,
                       dev_size=0.1, policy='project', labels=None):
    """Giữ test DrugNER; chia 10% tài liệu mỗi nguồn train thành dev."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = LABELS if labels is None else labels
    files = {'train': [], 'dev': [], 'test': []}
    for source in ('DrugBank', 'MedLine'):
        folder = data_dir / 'Train' / source
        if not list(folder.rglob('*.xml')):
            raise FileNotFoundError(f'Không tìm thấy XML: {folder}')
        train_files, dev_files = split_xml_files(folder, dev_size, seed)
        files['train'].extend(train_files)
        files['dev'].extend(dev_files)
    files['test'] = sorted((data_dir / 'Test' / 'Test for DrugNER task').rglob('*.xml'))
    if not files['test']:
        raise FileNotFoundError('Không tìm thấy XML của Test for DrugNER task')
    examples = {name: load_examples_from_files(paths) for name, paths in files.items()}
    prepared = audit_splits(examples, {label: i for i, label in enumerate(labels)},
                            output_dir / 'annotation_report.json', policy)
    manifest = dict(seed=seed, unit='xml_document', dev_fraction=dev_size,
                    **{name: [str(p) for p in paths] for name, paths in files.items()})
    (output_dir / 'split_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    loaders = {name: make_loader(rows, batch_size, name == 'train', seed)
               for name, rows in prepared.items()}
    return loaders, prepared, manifest
