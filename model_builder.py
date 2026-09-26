"""Frozen embeddings → source attention → BiLSTM (32) → CRF."""
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

class WordPieceSource(nn.Module):
    """Only BERT's input embedding table, never the Transformer encoder."""
    def __init__(self, model_name, device, trainable=False):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        pretrained = AutoModel.from_pretrained(model_name)
        self.embedding = pretrained.get_input_embeddings()
        del pretrained
        self.embedding.requires_grad_(trainable).to(device)
        self.dim = self.embedding.embedding_dim

    def forward(self, sentences):
        results = []
        for words in sentences:
            encoded = self.tokenizer(
                        words,
                        is_split_into_words=True,
                        add_special_tokens=False,
                        return_tensors='pt',
                        truncation=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                        verbose=False,
                    )
            pieces = self.embedding(encoded['input_ids'].to(self.embedding.weight.device))[0]
            positions = [[] for _ in words]
            for j, word_id in enumerate(encoded.word_ids()):
                if word_id is not None:
                    positions[word_id].append(j)
            if not positions or any(not indices for indices in positions):
                raise ValueError('Câu rỗng hoặc tokenizer làm mất token')
            results.append(torch.stack([pieces[indices].mean(0) for indices in positions]))
        return results
class FastTextSource(nn.Module):
    """Train-vocabulary table initialized from FastText; OOV uses frozen subwords."""
    def __init__(self, model_path, vocab, trainable=False):
        super().__init__()
        import fasttext
        import numpy as np
        self.fasttext = fasttext.load_model(str(model_path))
        self.dim = self.fasttext.get_dimension()
        self.vocab = list(vocab)
        if not self.vocab or len(set(self.vocab)) != len(self.vocab):
            raise ValueError('FastText vocabulary phải khác rỗng và không trùng')
        self.word2id = {word: i for i, word in enumerate(self.vocab)}
        initial = torch.from_numpy(np.stack([
            self.fasttext.get_word_vector(word) for word in self.vocab
        ])).float()
        self.embedding = nn.Embedding.from_pretrained(initial, freeze=not trainable)

    def forward(self, sentences):
        results = []
        device = self.embedding.weight.device
        for words in sentences:
            rows = []
            for word in words:
                word = word.lower()
                index = self.word2id.get(word)
                if index is None:
                    # Do not grow the vocabulary using dev/test. Preserve FastText OOV support.
                    vector = torch.as_tensor(self.fasttext.get_word_vector(word),
                                             dtype=self.embedding.weight.dtype, device=device)
                else:
                    vector = self.embedding(torch.tensor(index, device=device))
                rows.append(vector)
            results.append(torch.stack(rows))
        return results
class FlairSource(nn.Module):
    def __init__(self, names, device, trainable=False):
        super().__init__()
        import flair
        from flair.embeddings import FlairEmbeddings, StackedEmbeddings
        flair.device = device
        self.trainable = trainable
        self.model = StackedEmbeddings([
            FlairEmbeddings(name, fine_tune=trainable) for name in names
        ])
        self.model.requires_grad_(trainable).to(device)
        self.dim = self.model.embedding_length
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if not self.trainable:
            self.model.eval()
        return self

    def forward(self, words_batch):
        import flair
        from flair.data import Sentence
        flair.device = next(self.model.parameters()).device
        # Fresh objects each forward: no stale embeddings or autograd graphs.
        sentences = [Sentence(words) for words in words_batch]
        track_grad = self.trainable and torch.is_grad_enabled()
        # Flair can enable gradients internally; explicitly disable fine_tune for evaluation.
        embeddings = list(self.model.embeddings)
        previous = [embedding.fine_tune for embedding in embeddings]
        try:
            for embedding in embeddings:
                embedding.fine_tune = track_grad
            with torch.set_grad_enabled(track_grad):
                self.model.embed(sentences)
                results = [torch.stack([token.embedding for token in sentence])
                           for sentence in sentences]
        finally:
            for embedding, fine_tune in zip(embeddings, previous):
                embedding.fine_tune = fine_tune
            for sentence in sentences:
                sentence.clear_embeddings()
        return results
class SourceAttention(nn.Module):
    def __init__(self, source_dims, fusion_dim):
        super().__init__()
        if not source_dims:
            raise ValueError('Cần ít nhất một nguồn embedding')
        self.names = list(source_dims)
        self.projections = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim, fusion_dim),
                                         nn.LayerNorm(fusion_dim)) for name, dim in source_dims.items()})
        self.source_bias = nn.Parameter(torch.zeros(len(self.names), fusion_dim))
        self.score = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.Tanh(), nn.Linear(fusion_dim, 1, bias=False))

    def forward(self, features, mask):
        projected = torch.stack([self.projections[name](features[name]) for name in self.names], dim=2)
        weights = self.score(projected + self.source_bias).squeeze(-1).softmax(dim=2)
        weights = weights.masked_fill(~mask.unsqueeze(-1), 0)
        return (weights.unsqueeze(-1) * projected).sum(2), weights
class FusionBiLSTMCRF(nn.Module):
    def __init__(self, sources, source_dims, num_tags, fusion_dim=200, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.sources = nn.ModuleDict(sources)
        self.fusion = SourceAttention(source_dims, fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(fusion_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, batch, return_attention=False):
        features = {}
        for name, source in self.sources.items():
            vectors = source(batch['words'])
            features[name] = pad_sequence(vectors, batch_first=True)
        fused, weights = self.fusion(features, batch['mask'])
        packed = pack_padded_sequence(self.dropout(fused), batch['lengths'].cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=batch['mask'].size(1))
        emissions = self.classifier(self.dropout(output))
        return (emissions, weights) if return_attention else emissions

    def loss(self, batch):
        return -self.crf(self(batch), batch['tags'], mask=batch['mask'], reduction='mean')

    def decode(self, batch):
        return self.crf.decode(self(batch), mask=batch['mask'])


def build_sources(feature_config, device, fasttext_path, download_fasttext=True):
    """Khôi phục đúng thứ tự nguồn và vocabulary từ cấu hình checkpoint."""
    sources = {}
    for name in feature_config['sources']:
        trainable = feature_config['trainable_sources'][name]
        if name == 'biomedbert':
            sources[name] = WordPieceSource(feature_config['transformer'], device, trainable)
        elif name == 'flair':
            sources[name] = FlairSource(feature_config['flair'], device, trainable)
        elif name == 'fasttext':
            path = Path(fasttext_path)
            if not path.is_file():
                if not download_fasttext:
                    raise FileNotFoundError(path)
                from huggingface_hub import hf_hub_download
                path = Path(hf_hub_download('facebook/fasttext-en-vectors', 'model.bin',
                                           local_dir=str(path.parent)))
            sources[name] = FastTextSource(path, feature_config['fasttext_vocab'], trainable)
        else:
            raise ValueError(f'Unknown embedding source: {name}')
    return sources


def build_optimizer(model, lr=1e-3, weight_decay=1e-3):
    embedding_lrs = {'biomedbert': 1e-4, 'flair': 1e-5, 'fasttext': 1e-4}
    head = [p for name, p in model.named_parameters()
            if not name.startswith('sources.') and p.requires_grad]
    groups = [dict(params=head, lr=lr)]
    for name, source in model.sources.items():
        params = [p for p in source.parameters() if p.requires_grad]
        if params:
            groups.append(dict(params=params, lr=embedding_lrs[name]))
    return torch.optim.AdamW(groups, weight_decay=weight_decay)
