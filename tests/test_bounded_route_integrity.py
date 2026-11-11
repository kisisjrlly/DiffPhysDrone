import json
from pathlib import Path
import pytest
from bounded_route.integrity import file_hash, verify_prepared, verify_reader_metadata, verify_checkpoint


def fixture_files(out):
    (out/'reader.pth').write_bytes(b'frozen-reader')
    (out/'reader_data.pth').write_bytes(b'frozen-data')
    split=dict(training=list(range(96)),validation=list(range(96,128)),data_hash=file_hash(out/'reader_data.pth'))
    (out/'reader_split.json').write_text(json.dumps(split))
    (out/'prepared.json').write_text(json.dumps(dict(reader_hash=file_hash(out/'reader.pth'),data_hash=file_hash(out/'reader_data.pth'))))
    return dict(data_hash=file_hash(out/'reader_data.pth'),split_hash=file_hash(out/'reader_split.json'))


def test_matching_frozen_artifacts_and_checkpoint_are_reusable(tmp_path):
    meta=fixture_files(tmp_path)
    verify_prepared(tmp_path);verify_reader_metadata(tmp_path,meta)
    verify_checkpoint(tmp_path,dict(reader_hash=file_hash(tmp_path/'reader.pth'),config_hash='config'),'config')


@pytest.mark.parametrize('name',['reader.pth','reader_data.pth'])
def test_mutated_frozen_artifact_rejected(tmp_path,name):
    fixture_files(tmp_path);(tmp_path/name).write_bytes(b'different')
    with pytest.raises(RuntimeError,match='Frozen artifact mismatch'):verify_prepared(tmp_path)


def test_mutated_whole_trajectory_split_rejected(tmp_path):
    meta=fixture_files(tmp_path)
    data=json.loads((tmp_path/'reader_split.json').read_text());data['validation'][0]=0
    (tmp_path/'reader_split.json').write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match='split mismatch'):verify_prepared(tmp_path)
    with pytest.raises(RuntimeError,match='trajectory split'):verify_reader_metadata(tmp_path,meta)


@pytest.mark.parametrize('change',['reader_hash','config_hash'])
def test_mismatched_optimizer_checkpoint_rejected(tmp_path,change):
    fixture_files(tmp_path)
    state=dict(reader_hash=file_hash(tmp_path/'reader.pth'),config_hash='config');state[change]='different'
    with pytest.raises(RuntimeError):verify_checkpoint(tmp_path,state,'config')
