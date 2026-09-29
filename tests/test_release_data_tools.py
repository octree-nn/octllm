"""Check data joins, held-out exclusion, and render names used by public commands."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset_toolkits.caption.export_captions import export_captions
from dataset_toolkits.render.render_mesh_subdirs import _build_output_prefix


def run_tool(script, *args):
    result = subprocess.run(
        [sys.executable, str(ROOT / script), *map(str, args)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_exported_captions_join_tokens_and_images_into_three_tasks(tmp_path):
    captions = tmp_path / 'captions'
    tokens = tmp_path / 'tokens'
    renders = tmp_path / 'renders'
    output = tmp_path / 'sft'
    captions.mkdir()
    tokens.mkdir()
    (renders / 'chair').mkdir(parents=True)
    (renders / 'chair' / '014.png').write_bytes(b'filename-only test fixture')
    (tokens / 'chair_bytes.txt').write_text('0 128 255')
    (tokens / 'unpaired_bytes.txt').write_text('255')
    (captions / 'chair.txt').write_text('A chair, with a curved back.\nFour legs.')
    (captions / 'empty.txt').write_text('  ')
    csv_path = tmp_path / 'captions.csv'
    assert export_captions(captions, csv_path) == 1
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{'asset_id': 'chair', 'caption': 'A chair, with a curved back.\nFour legs.'}]
    run_tool('dataset_toolkits/prepare_trellis_sft.py',
             '--token_dir', tokens, '--render_dir', renders,
             '--caption_csv', csv_path, '--output_dir', output)
    for task in ('description', 'understanding', 'image'):
        rows = json.loads((output / f'llm_trellis_dataset_1_{task}.json').read_text())
        assert len(rows) == 1
        sample = rows[0]
        assert [m['role'] for m in sample['messages']] == ['user', 'assistant']
        mesh_message = sample['messages'][0 if task == 'understanding' else 1]['content']
        assert '<mesh_bos><mesh0><mesh128><mesh255><mesh_eos>' in mesh_message
        assert '#response#' not in json.dumps(sample)
        assert '#object_name#' not in json.dumps(sample)
        if task == 'image':
            assert sample['images'] == [str(renders / 'chair' / '014.png')]


@pytest.mark.parametrize('task', ['description', 'understanding', 'image'])
def test_shapenet_public_cli_excludes_fixed_test_assets(tmp_path, task):
    tokens = tmp_path / 'tokens'
    renders = tmp_path / 'renders'
    output = tmp_path / 'sft'
    for name, code in [('train_asset', '128'), ('held_out', '255')]:
        (tokens / name).mkdir(parents=True)
        (tokens / name / f'{name}_bytes.txt').write_text(code)
        (renders / name).mkdir(parents=True)
        (renders / name / '014.png').write_bytes(b'filename-only test fixture')
    descriptions = tmp_path / 'descriptions.json'
    descriptions.write_text(json.dumps([
        {'name': 'train_asset', 'description': 'A narrow chair.'},
        {'name': 'held_out', 'description': 'A wide chair.'},
    ]))
    split = tmp_path / 'test.json'
    split.write_text(json.dumps([{'name': 'held_out'}]))
    run_tool('dataset_toolkits/prepare_shapenet_sft.py',
             '--category', 'chair', '--task', task,
             '--description-json', descriptions, '--token-dir', tokens,
             '--render-dir', renders, '--test-json', split, '--output-dir', output)
    kind = 'mllm' if task == 'image' else 'llm'
    rows = json.loads((output / f'{kind}_shapenet_dataset_chair_1_{task}.json').read_text())
    assert len(rows) == 1
    content = json.dumps(rows)
    assert '<mesh128>' in content
    assert '<mesh255>' not in content
    assert not list(output.glob('*_test_*'))


def test_octllm_render_names_match_toys4k_metric_inputs(tmp_path):
    source = tmp_path / 'image_to_3d'
    mesh = source / 'asset_1' / 'asset_1.glb'
    assert _build_output_prefix(mesh, source, 'octllm') + '__014.png' == 'asset_1__glb__014.png'
    assert _build_output_prefix(mesh, source) == 'asset_1__asset_1__glb'
    with pytest.raises(ValueError, match='Expected'):
        _build_output_prefix(source / 'asset_1' / 'other.glb', source, 'octllm')


def test_condition_renderer_passes_explicit_blender_to_worker(tmp_path):
    meshes = tmp_path / 'meshes'
    meshes.mkdir()
    (meshes / 'chair.obj').write_text('v 0 0 0\n')
    output = tmp_path / 'renders'
    fake_blender = tmp_path / 'blender'
    fake_blender.write_text(
        f'#!{sys.executable}\n'
        'import json, sys\n'
        'from pathlib import Path\n'
        'args = sys.argv\n'
        'out = Path(args[args.index("--output_folder") + 1])\n'
        'out.mkdir(parents=True, exist_ok=True)\n'
        '(out / "invocation.json").write_text(json.dumps(args))\n'
    )
    fake_blender.chmod(0o755)
    run_tool('dataset_toolkits/render/render_cond.py',
             '--input_dir', meshes, '--output_dir', output,
             '--blender-path', fake_blender, '--max_workers', '1')
    args = json.loads((output / 'chair' / 'invocation.json').read_text())
    render_script = Path(args[args.index('-P') + 1])
    assert render_script == ROOT / 'dataset_toolkits/render/blender_script/render.py'
    assert render_script.is_file()
    assert args[args.index('--engine') + 1] == 'BLENDER_EEVEE'
