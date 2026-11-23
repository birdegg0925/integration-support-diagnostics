#!/usr/bin/env python3
"""Download one explicitly pinned public model; no account or paid inference."""
import argparse, json
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--revision', default='7ae557604adf67be50417f59c2c2f167def9a775')
    args = parser.parse_args()
    target = Path(args.out)
    snapshot_download(args.model, revision=args.revision, local_dir=target,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])
    info = {'model_id': args.model, 'revision': args.revision,
            'source': 'https://huggingface.co/' + args.model,
            'weights_bytes': sum(p.stat().st_size for p in target.glob('*.safetensors'))}
    (target / 'pinned_model_origin.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == '__main__':
    main()
