import json
import struct
import torch
import pytest
from safetensors.torch import save_file
from ollm.bounded.budget import MemoryBudget, MemoryBudgetError
from ollm.bounded.checkpoint import TensorStore, WeightRef
from ollm.bounded.cache import DiskState
from ollm.bounded.model import BoundedModel


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.float16])
def test_cache_isolated_incremental_and_exact(tmp_path,dtype):
    root=tmp_path/'cache';root.mkdir();sentinel=root/'keep';sentinel.write_text('not ours')
    a,b=DiskState(root,max_block_tokens=3),DiskState(root,max_block_tokens=3)
    x=torch.arange(24).reshape(6,2,2).to(dtype)
    for cache in (a,b):
        cache.append((7,'k'),x[:3],position=0)
        cache.append((7,'k'),x[3:],position=3)
        torch.testing.assert_close(cache.read((7,'k'),2,5),x[2:5])
        assert cache.length((0,'k'))==0
        with pytest.raises(ValueError):cache.append((7,'k'),x[:1],position=0)
        with pytest.raises(ValueError):cache.read((7,'k'),0,6)
    a.close();assert b.path.exists() and sentinel.exists()
    b.close();assert list(root.iterdir())==[sentinel]
    with pytest.raises(RuntimeError):a.length((7,'k'))


def test_recurrent_replace_no_leak_and_quota(tmp_path):
    with DiskState(tmp_path,max_bytes=128) as cache:
        cache.save_state((12,'recurrent'),torch.ones(4))
        for n in range(5):cache.save_state((12,'recurrent'),torch.full((4,),float(n)))
        assert cache.bytes_stored==16
        torch.testing.assert_close(cache.state((12,'recurrent')),torch.full((4,),4.))
        with pytest.raises(RuntimeError,match='quota'):cache.save_state((12,'recurrent'),torch.ones(100))
        with pytest.raises(ValueError):cache.save_state((12,'k'),torch.ones(1))
    assert not list(tmp_path.iterdir())


def test_cache_exception_cleanup(tmp_path):
    with pytest.raises(RuntimeError):
        with DiskState(tmp_path) as cache:
            cache.append((0,'k'),torch.ones(1,3),position=0)
            raise RuntimeError('test')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_tensor_rows_and_expert_slice(tmp_path,dtype):
    x=torch.arange(120).reshape(3,8,5).to(dtype)
    save_file({'bank':x},str(tmp_path/'model.safetensors'))
    store=TensorStore(tmp_path,max_read_bytes=80)
    ref=WeightRef('bank',expert=2,row_start=2,row_stop=6)
    assert store.shape(ref)==(4,5)
    torch.testing.assert_close(store.rows(ref,0,2,dtype=dtype),x[2,2:4])
    assert store.max_read_observed<=80
    with pytest.raises(MemoryBudgetError):store.small('bank')
    with pytest.raises(ValueError):store.rows(ref,0,100)


@pytest.mark.parametrize('start,stop',[(0,10),(120,140),(128,256),(256,260)])
def test_fp8_partial_block_dequantization(tmp_path,start,stop):
    torch.manual_seed(7)
    q=(torch.randn(260,257)*.1).to(torch.float8_e4m3fn)
    s=torch.rand(3,3)+.2
    save_file({'x.weight':q,'x.weight_scale_inv':s},str(tmp_path/'model.safetensors'))
    store=TensorStore(tmp_path,max_read_bytes=1024**2)
    out=store.rows('x.weight',start,stop,dtype=torch.float32)
    full=q.float()*s.repeat_interleave(128,0).repeat_interleave(128,1)[:260,:257]
    torch.testing.assert_close(out,full[start:stop])


def test_fp8_missing_scales_rejected(tmp_path):
    save_file({'x.weight':torch.ones(8,8).to(torch.float8_e4m3fn)},str(tmp_path/'model.safetensors'))
    with pytest.raises(ValueError,match='scales'):TensorStore(tmp_path).rows('x.weight')


def test_shard_escape_and_bad_index(tmp_path):
    outside=tmp_path/'outside.safetensors';save_file({'x':torch.ones(1)},str(outside))
    root=tmp_path/'model';root.mkdir()
    index=root/'model.safetensors.index.json'
    index.write_text(json.dumps({'weight_map':{'x':'../outside.safetensors'}}))
    with pytest.raises(ValueError,match='escapes'):TensorStore(root)
    (root/'model.safetensors').write_bytes(outside.read_bytes())
    index.write_text(json.dumps({'weight_map':{'missing':'model.safetensors'}}))
    with pytest.raises(ValueError,match='match'):TensorStore(root)


@pytest.mark.parametrize('header',[
    {'x':{'dtype':'F32','shape':[2],'data_offsets':[0,4]}},
    {'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]},'y':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}},
    {'x':{'dtype':'UNKNOWN','shape':[1],'data_offsets':[0,4]}},
])
def test_invalid_tensor_headers(tmp_path,header):
    data=json.dumps(header).encode()
    (tmp_path/'model.safetensors').write_bytes(struct.pack('<Q',len(data))+data+b'\0'*8)
    with pytest.raises(ValueError):TensorStore(tmp_path)


def test_duplicate_json_and_truncated_file(tmp_path):
    data=b'{"x":{},"x":{}}'
    (tmp_path/'model.safetensors').write_bytes(struct.pack('<Q',len(data))+data)
    with pytest.raises(ValueError,match='Duplicate'):TensorStore(tmp_path)
    (tmp_path/'model.safetensors').write_bytes(b'abc')
    with pytest.raises(ValueError,match='Truncated'):TensorStore(tmp_path)


@pytest.mark.parametrize('kwargs',[{'vram_bytes':8_000_000_000},{'prefill_tokens':0},{'max_output_tokens':True},
    {'attention_block_tokens':-1},{'workspace_bytes':7_000_000_000}])
def test_invalid_budget(kwargs):
    with pytest.raises(ValueError):MemoryBudget(**kwargs)


def test_scalar_and_empty_state_roundtrip(tmp_path):
    save_file({'scalar':torch.tensor(2.)},str(tmp_path/'model.safetensors'))
    assert TensorStore(tmp_path).small('scalar').shape==()
    with DiskState(tmp_path) as state:
        state.save_state((0,'conv'),torch.empty(4,0))
        assert state.state((0,'conv')).shape==(4,0)
