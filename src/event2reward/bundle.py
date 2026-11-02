"""Model bundles: weights, processor files and a manifest with file hashes.

This is the format of the Hugging Face model repositories.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

FORMAT = "event2reward.bundle.v2"
SIDECARS = ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
            "video_preprocessor_config.json", "chat_template.jinja", "merges.txt", "vocab.json", "LICENSE")


def asset_path(root, name):
    """Resolve declared files, including the Hub's snapshots-to-blobs links."""
    relative = PurePosixPath(name)
    if (relative.is_absolute() or relative.as_posix() != name
            or ".." in relative.parts or "\\" in name or not relative.parts):
        raise ValueError(f'invalid bundle file path: {name}')
    link = root/name
    resolved = link.resolve()
    if not resolved.is_relative_to(root):
        # snapshot_download links each file into the repository's blobs/
        # directory; recent huggingface_hub versions link that blob once more
        # into the cache-wide hub/blobs/. Accept only those Hub cache targets.
        hub = root.parent.parent
        in_hub_cache = (root.parent.name == 'snapshots' and link.is_symlink()
                        and (resolved.parent == hub/'blobs'
                             or resolved.is_relative_to(hub.parent/'blobs')))
        if not in_hub_cache:
            raise ValueError(f'bundle file escapes directory: {name}')
    return resolved


def is_doc_asset(name):
    """Model card and license texts: shipped with a bundle, but not part of the model."""
    return name in {'README.md', 'LICENSE', 'NOTICE'} or name.startswith('LICENSES/')


def validate_destination(destination, sources):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if any(destination.is_relative_to(Path(source).resolve()) for source in sources):
        raise ValueError('destination must not be inside a source bundle or package directory')
    return destination


def file_records(root):
    return {str(p.relative_to(root)): {'bytes':p.stat().st_size, 'sha256':sha256(p)}
            for p in sorted(root.rglob('*')) if p.is_file() and p != root/'event2reward.json'}


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read_manifest(path, *, verify=True):
    root=Path(path).resolve()
    manifest=json.loads((root/'event2reward.json').read_text())
    if manifest.get('format') != FORMAT:
        raise ValueError(f'unsupported bundle format {manifest.get("format")!r}; expected {FORMAT}')
    if manifest.get('num_bins')!=101 or manifest.get('history_frames')!=5:
        raise ValueError('unsupported bundle: expected 101 progress bins and 5 history frames')
    for name,record in manifest['files'].items():
        if is_doc_asset(name):
            continue  # the model card and notices can be edited without invalidating the model
        file=asset_path(root,name)
        if not file.is_file() or file.stat().st_size!=record['bytes']:
            raise ValueError(f'missing or truncated bundle file: {name}')
        if verify and sha256(file)!=record['sha256']:
            raise ValueError(f'bundle hash mismatch: {name}')
    for required in ('model/model.safetensors', 'state.json',
                     *(f'backbone/{name}' for name in SIDECARS)):
        if required not in manifest['files']:raise ValueError(f'unregistered file: {required}')
    return manifest


def load_model(path, *, attention='sdpa', verify=True, loss=None, torch_dtype=None,
               validate_query_tokens=True):
    import torch
    import transformers
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from event2reward.modeling.model import ProgressModelVLM
    from event2reward.modeling.qwen import LoadedModel, PROGRESS_QUERY, TREND_QUERY, NULL_GOAL
    root=Path(path);m=read_manifest(root,verify=verify)
    if attention not in ('sdpa','eager'):raise ValueError('supported attention: sdpa or eager')
    dtype=torch.bfloat16 if torch_dtype is None else torch_dtype
    processor=transformers.AutoProcessor.from_pretrained(root/'backbone',local_files_only=True,trust_remote_code=False)
    for name,token in [('progress',PROGRESS_QUERY),('trend',TREND_QUERY),('null_goal',NULL_GOAL)]:
        if processor.tokenizer.convert_tokens_to_ids(token)!=m['query_token_ids'][name]:
            raise ValueError(f'wrong {name} token ID in exported processor')
    if len(processor.tokenizer)!=m['vocab_size']:raise ValueError('wrong exported vocabulary size')
    config=transformers.Qwen3_5Config.from_pretrained(root/'backbone',local_files_only=True)
    config.text_config.vocab_size=m['vocab_size']
    config._attn_implementation=attention
    # Persistent parameters are populated strictly from the complete fine-tuned state.
    # Nonpersistent rotary buffers are initialized normally, not left on meta.
    with init_empty_weights():
        backbone=transformers.Qwen3_5Model(config)
        model=ProgressModelVLM(backbone,hidden_size=m['hidden_size'],
            progress_query_token_id=m['query_token_ids']['progress'],
            trend_query_token_id=m['query_token_ids']['trend'],num_bins=m['num_bins'],
            head_dropout=m['head_dropout'],validate_query_tokens=validate_query_tokens,loss=loss)
    state=load_file(str(root/'model/model.safetensors'),device='cpu')
    model.load_state_dict(state,strict=True,assign=True)
    del state
    if any(p.is_meta for p in model.parameters()):raise RuntimeError('uninitialized model parameters')
    # Do not cast rotary/nonpersistent buffers: the source HF loader retains
    # these in float32. Module.to(bfloat16) changes the position computation.
    for parameter in model.parameters():
        if parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype=dtype)
    return LoadedModel(model,processor),m


def save_instance(model, processor, source, destination, *, updates=None, state=None):
    """Save current in-memory parameters, not a copy of the initial checkpoint."""
    from safetensors.torch import save_model
    source=Path(source).resolve()
    destination=validate_destination(destination,[source,Path(__file__).parent])
    manifest=read_manifest(source)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix=destination.name+'.partial-',dir=destination.parent))
    try:
        for name in manifest['files']:
            if name == 'model/model.safetensors':
                continue
            if is_doc_asset(name) and not (source/name).is_file():
                continue  # a removed document is not carried over
            if name == 'state.json' and state is not None:
                continue
            target = temp/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/name, target)
        if state is not None:
            (temp/'state.json').write_text(json.dumps(state, indent=2)+'\n')
        (temp/'model').mkdir(exist_ok=True)
        save_model(model,str(temp/'model/model.safetensors'),metadata={'format':'pt'})
        processor.save_pretrained(temp/'backbone')
        manifest['serialization']='current_in_memory_parameters'
        manifest['parent_bundle_manifest_sha256']=sha256(source/'event2reward.json')
        updated=file_records(temp)
        manifest.update(updates or {})
        manifest['files']=updated
        (temp/'event2reward.json').write_text(json.dumps(manifest,indent=2)+'\n')
        read_manifest(temp)
        os.rename(temp,destination)
    except BaseException:
        shutil.rmtree(temp)
        raise
    return destination
