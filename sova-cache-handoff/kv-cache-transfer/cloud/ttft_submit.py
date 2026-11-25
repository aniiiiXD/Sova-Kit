"""Measure TTFT with the exact archived 1.7B-to-8B mapper."""
import argparse
import json
from pathlib import Path

from .harvey_pilot_submit import INSTANCE, PAIR, ROOT, dispatch, sha
from .package_integrity import verify_worker_imports
from .submit import BUCKET, aws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instance', choices=(INSTANCE, 'ml.g6e.4xlarge'), default=INSTANCE)
    blocks = parser.add_mutually_exclusive_group()
    blocks.add_argument('--preserve-gemm-blocks', dest='fast_blocks', action='store_const', const=256,
                        help='Keep reference GEMM blocks and compare cache tensors (default)')
    blocks.add_argument('--experimental-large-blocks', dest='fast_blocks', action='store_const', const=8192,
                        help='Reproduce the faster variant that failed strict parity')
    parser.set_defaults(fast_blocks=256)
    parser.add_argument('--training-run', type=Path,
                        help='Independently verify a completed answer-training run before timing it')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()

    verified = None
    if args.training_run:
        if args.fast_blocks != 256:
            raise ValueError('A trained candidate requires the verified 256-token projection path')
        from scripts.verify_answer_mapper_run import verify_run
        verified = verify_run(args.training_run)
        if verified['verdict'] != 'ready_for_timing':
            raise ValueError('Candidate did not pass the independently reproduced development gate')
        parent = json.loads((args.training_run/'training/submission.json').read_text())
        if parent['config']['mode'] != 'answer_train' or parent['config']['models'] != PAIR:
            raise ValueError('Unexpected answer-training parent')
        origin = f"s3://{BUCKET}/kv-transfer/{verified['job_name']}/live/best/"
    else:
        parent = json.loads((ROOT/'reports/quality-8b-v1/submission.json').read_text())
        if parent['config']['models'] != PAIR or parent['config']['benchmark'] != 'quality':
            raise ValueError('Unexpected archived model pair or benchmark')
        name = parent['job_name']
        state = json.loads(aws('sagemaker', 'describe-training-job', '--training-job-name', name))
        if state['TrainingJobStatus'] != 'Completed':
            raise ValueError('Expected completed parent job')
        origin = f's3://{BUCKET}/kv-transfer/{name}/live/ridge/'
    manifest = json.loads(aws('s3', 'cp', origin + 'manifest.json', '-', '--only-show-errors'))
    if (manifest.get('mapper_format') != 'affine-cache-v1' or manifest.get('head_mixing') is not True
            or manifest.get('source_geometry') != [8, 128, 28]
            or manifest.get('target_geometry') != [8, 128, 36]):
        raise ValueError('Unsupported historical mapper')

    native_prefix = verified['native_prefix_tokens'] if verified else 0
    if verified and (manifest['weights_sha256'] != verified['weights_sha256']
                     or manifest['native_prefix_policy'] != str(native_prefix)):
        raise ValueError('Timing manifest differs from independently verified checkpoint')
    dataset = ROOT/'datasets/loop-holdout-v1'
    reference = json.loads((dataset/'manifest.json').read_text())
    inputs = {'inputs/quality.jsonl': dataset/'quality.jsonl'}
    if sha(inputs['inputs/quality.jsonl']) != reference['files']['quality.jsonl']['sha256']:
        raise ValueError('Timing dataset hash differs')
    files = {path.relative_to(ROOT).as_posix(): path for folder in ('kvverify', 'cloud')
             for path in (ROOT/folder).glob('*.py')}
    files['cloud/recovery_bootstrap.sh'] = ROOT/'cloud/recovery_bootstrap.sh'
    files.update(inputs)
    config = {'mode': 'ttft', 'models': PAIR, 'seed': 19, 'documents': 8, 'repeats': 5,
              'max_runtime_seconds': 2400, 'worker_timeout_seconds': 2040,
              'fast_block_tokens': args.fast_blocks,
              'diagnose_cache_tensors': args.fast_blocks == 256 and not verified,
              'native_prefix_tokens': native_prefix,
              'verified_training_job': verified['job_name'] if verified else None,
              'mapper_origin': origin, 'input_mapper_dir': '/opt/ml/input/data/mapper',
              'mapper_weights_sha256': manifest['weights_sha256'],
              'input_sha256': {path: sha(source) for path, source in inputs.items()},
              'verified_worker_imports': verify_worker_imports(files, 'cloud/ttft_worker.py')}
    dispatch(config, files, instance=args.instance, output=args.output, submit=args.submit, job_kind='ttft',
             mapper_input=True)


if __name__ == '__main__':
    main()
