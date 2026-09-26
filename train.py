"""Chạy: python train.py; đánh giá trọng số có sẵn: python train.py --eval-only.

Dependencies: torch, scikit-learn, seqeval, tqdm, pytorch-crf, transformers,
flair, fasttext-wheel, huggingface_hub. Embedding pretrained được tải khi cần.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models' / 'best _no_grad_32.pt')
    parser.add_argument('--save-path', type=Path, default=ROOT / 'models' / 'best_modular.pt')
    parser.add_argument('--fasttext-path', type=Path, default=ROOT / 'models' / 'pretrained' / 'fasttext-en-vectors' / 'model.bin')
    parser.add_argument('--no-download-fasttext', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=23022006)
    parser.add_argument('--device', default=None, help='cpu, cuda, cuda:0; mặc định tự chọn')
    parser.add_argument('--annotation-policy', choices=['project', 'strict'], default='project')
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    import data_setup
    import engine
    import model_builder
    import utils

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    checkpoint = utils.load_checkpoint(args.checkpoint) if args.eval_only else None
    seed = checkpoint.get('seed', args.seed) if checkpoint else args.seed
    utils.set_seed(seed)
    labels = checkpoint['labels'] if checkpoint else data_setup.LABELS
    policy = checkpoint['feature_config']['annotation_policy'] if checkpoint else args.annotation_policy
    loaders, prepared, manifest = data_setup.create_dataloaders(
        args.data_dir, args.output_dir, args.batch_size, seed,
        policy=policy, labels=labels)
    print('Số câu:', {name: len(rows) for name, rows in prepared.items()})
    feature_config = checkpoint['feature_config'] if checkpoint else dict(
        annotation_policy=policy, sources=['biomedbert', 'flair', 'fasttext'],
        transformer='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
        flair=['pubmed-forward', 'pubmed-backward'], fasttext=str(args.fasttext_path),
        token_pattern=data_setup.TOKEN_PATTERN, biomedbert_representation='input_embeddings',
        trainable_sources=dict(biomedbert=False, flair=False, fasttext=False),
        fasttext_representation='train_vocab_table_with_frozen_oov',
        fasttext_vocab=sorted({w.lower() for row in prepared['train'] for w in row['words']}),
        pooling='mean_wordpiece_lookup', fasttext_lowercase=True,
        alignment='regex_exact_boundaries_v1')
    sources = model_builder.build_sources(feature_config, device, args.fasttext_path,
                                          not args.no_download_fasttext)
    model_config = checkpoint['model_config'] if checkpoint else dict(
        source_dims={name: source.dim for name, source in sources.items()},
        num_tags=len(labels), fusion_dim=200, hidden_dim=32, dropout=0.5)
    model = model_builder.FusionBiLSTMCRF(sources=sources, **model_config).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        optimizer = model_builder.build_optimizer(model, args.lr)
        engine.train(model, loaders['train'], loaders['dev'], optimizer, labels, device,
                     model_config, feature_config, manifest, args.output_dir, args.save_path,
                     args.epochs, args.patience, seed)
        checkpoint = utils.load_checkpoint(args.save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    metrics = engine.evaluate_model(model, loaders['test'], labels, device)
    print('Test (best checkpoint):', json.dumps(metrics, indent=2))
    (args.output_dir / 'test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
