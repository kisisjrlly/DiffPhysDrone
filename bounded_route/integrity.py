"""Reject cross-run reuse before loading optimizers or frozen reader weights."""
import hashlib
import json
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_prepared(out):
    out=Path(out);path=out/'prepared.json'
    if not path.exists():return
    prepared=json.loads(path.read_text())
    for name,key in [('reader.pth','reader_hash'),('reader_data.pth','data_hash')]:
        target=out/name
        if not target.is_file() or file_hash(target)!=prepared.get(key):
            raise RuntimeError(f'Frozen artifact mismatch: {name}; results cannot be reused')
    split=json.loads((out/'reader_split.json').read_text())
    if split.get('data_hash')!=prepared['data_hash']:
        raise RuntimeError('Reader split refers to different data')
    train,val=split.get('training',[]),split.get('validation',[])
    if len(train)!=96 or len(val)!=32 or set(train)&set(val) or set(train)|set(val)!=set(range(128)):
        raise RuntimeError('Reader trajectory split mismatch')


def verify_reader_metadata(out,metadata):
    out=Path(out)
    if metadata.get('data_hash')!=file_hash(out/'reader_data.pth'):
        raise RuntimeError('Reader weights refer to different data')
    if metadata.get('split_hash')!=file_hash(out/'reader_split.json'):
        raise RuntimeError('Reader weights refer to different trajectory split')


def verify_checkpoint(out,state,config_hash):
    out=Path(out)
    if state.get('reader_hash')!=file_hash(out/'reader.pth'):
        raise RuntimeError('Optimizer checkpoint refers to different reader weights')
    if state.get('config_hash')!=config_hash:
        raise RuntimeError('Optimizer checkpoint refers to different execution configuration')
