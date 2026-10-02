import math
import torch
import torch.nn.functional as F
import pytest

from ollm.bounded.budget import MemoryBudget, MemoryBudgetError
from ollm.bounded.model import load_bounded_model
from ollm.bounded.qwen3_vl import mrope


def _rotate_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


def _dense_merger(x, w, base, hidden, merge, out_hidden, postshuffle):
    group = hidden * merge * merge
    if postshuffle:
        z = x.reshape(-1, group)
        z = F.layer_norm(z, (group,), w[base+'norm.weight'], w[base+'norm.bias'], 1e-6)
    else:
        z = F.layer_norm(x, (hidden,), w[base+'norm.weight'], w[base+'norm.bias'], 1e-6)
        z = z.reshape(-1, group)
    z = F.linear(z, w[base+'linear_fc1.weight'], w[base+'linear_fc1.bias'])
    z = F.gelu(z)
    return F.linear(z, w[base+'linear_fc2.weight'], w[base+'linear_fc2.bias'])


def _dense_vision(pixel, grid, raw, w):
    v = raw['vision_config']; p = 'model.visual.'
    h, heads, merge = v['hidden_size'], v['num_heads'], v['spatial_merge_size']
    hd = h // heads
    x = F.linear(pixel, w[p+'patch_embed.proj.weight'].reshape(h, -1), w[p+'patch_embed.proj.bias'])

    # Independent bilinear interpolation of the square learned position table.
    side = int(math.isqrt(v['num_position_embeddings']))
    table = w[p+'pos_embed.weight'].reshape(side, side, h).permute(2, 0, 1)[None]
    pos_parts, coords_parts = [], []
    for t, gh, gw in grid.tolist():
        pos = F.interpolate(table, size=(gh, gw), mode='bilinear', align_corners=True)[0].permute(1, 2, 0)
        pos = pos.repeat(t, 1, 1, 1)
        pos = pos.view(t, gh//merge, merge, gw//merge, merge, h).permute(0,1,3,2,4,5).flatten(0,4)
        pos_parts.append(pos)
        rows = torch.arange(gh//merge)[:,None,None,None]*merge + torch.arange(merge)[None,None,:,None]
        cols = torch.arange(gw//merge)[None,:,None,None]*merge + torch.arange(merge)[None,None,None,:]
        rows = rows.expand(gh//merge,gw//merge,merge,merge).reshape(-1)
        cols = cols.expand(gh//merge,gw//merge,merge,merge).reshape(-1)
        coords_parts.append(torch.stack((rows,cols),-1).repeat(t,1))
    x = x + torch.cat(pos_parts)
    coords = torch.cat(coords_parts)
    inv = 10000.0 ** (-torch.arange(0, hd//2, 2, dtype=torch.float32)/(hd//2))
    freq = (coords.float()[:,:,None] * inv[None,None]).flatten(1)
    emb = torch.cat((freq,freq),-1)
    cos,sin=emb.cos()[:,None],emb.sin()[:,None]

    deep=[]
    for layer in range(v['depth']):
        b=p+f'blocks.{layer}.'
        z=F.layer_norm(x,(h,),w[b+'norm1.weight'],w[b+'norm1.bias'],1e-6)
        qkv=F.linear(z,w[b+'attn.qkv.weight'],w[b+'attn.qkv.bias']).reshape(len(z),3,heads,hd)
        q,k,val=qkv.unbind(1)
        q=q*cos+_rotate_half(q)*sin; k=k*cos+_rotate_half(k)*sin
        pieces=[]; start=0
        for t,gh,gw in grid.tolist():
            for _ in range(t):
                end=start+gh*gw
                score=torch.einsum('thd,shd->hts',q[start:end],k[start:end])/math.sqrt(hd)
                att=torch.einsum('hts,shd->thd',score.softmax(-1),val[start:end]).reshape(end-start,h)
                pieces.append(att); start=end
        att=torch.cat(pieces)
        x=x+F.linear(att,w[b+'attn.proj.weight'],w[b+'attn.proj.bias'])
        z=F.layer_norm(x,(h,),w[b+'norm2.weight'],w[b+'norm2.bias'],1e-6)
        z=F.gelu(F.linear(z,w[b+'mlp.linear_fc1.weight'],w[b+'mlp.linear_fc1.bias']),approximate='tanh')
        x=x+F.linear(z,w[b+'mlp.linear_fc2.weight'],w[b+'mlp.linear_fc2.bias'])
        if layer in v['deepstack_visual_indexes']:
            j=v['deepstack_visual_indexes'].index(layer)
            deep.append(_dense_merger(x,w,p+f'deepstack_merger_list.{j}.',h,merge,v['out_hidden_size'],True))
    final=_dense_merger(x,w,p+'merger.',h,merge,v['out_hidden_size'],False)
    return final,deep


def test_qwen3vl_streamed_vision_matches_dense_reference(make_checkpoint):
    path,raw,w=make_checkpoint('qwen3_vl')
    budget=MemoryBudget(max_visual_tokens=64,max_images=2,attention_block_tokens=3,weight_tile_bytes=4096)
    model=load_bounded_model(path,device='cpu',dtype=torch.float32,budget=budget)
    grid=torch.tensor([[1,4,6]])
    patch=raw['vision_config']['in_channels']*raw['vision_config']['temporal_patch_size']*raw['vision_config']['patch_size']**2
    torch.manual_seed(17); pixels=torch.randn(24,patch)
    actual,deep,n=model.encode_images(pixels,grid)
    expected,expected_deep=_dense_vision(pixels,grid,raw,w)
    assert n==24
    torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-5)
    assert len(deep)==len(expected_deep)
    for a,e in zip(deep,expected_deep):
        torch.testing.assert_close(a,e,atol=3e-5,rtol=3e-5)


def test_qwen3vl_mrope_reduces_to_text_rope_when_axes_equal(make_checkpoint):
    path,_,_=make_checkpoint('qwen3_vl')
    model=load_bounded_model(path,device='cpu',dtype=torch.float32,
                             budget=MemoryBudget(max_visual_tokens=16))
    x=torch.randn(7,2,model.config.head_dim)
    pos=torch.arange(7).repeat(3,1)
    from ollm.bounded.kernels import rotary
    torch.testing.assert_close(mrope(x,pos,model.config.rope),
                               rotary(x,torch.arange(7),model.config.head_dim,model.config.rope))


def test_qwen3vl_image_generation_and_bounds(make_checkpoint):
    path,raw,_=make_checkpoint('qwen3_vl')
    budget=MemoryBudget(max_visual_tokens=16,max_images=1,max_output_tokens=4)
    model=load_bounded_model(path,device='cpu',dtype=torch.float32,budget=budget)
    v=raw['vision_config']; patch=v['in_channels']*v['temporal_patch_size']*v['patch_size']**2
    pixels=torch.randn(16,patch); grid=torch.tensor([[1,4,4]])
    ids=torch.tensor([[1,raw['vision_start_token_id'],*([raw['image_token_id']]*4),raw['vision_end_token_id'],4]])
    out=model.generate(ids,pixel_values=pixels,image_grid_thw=grid,max_new_tokens=2,eos_token_id=[])
    assert out.shape[1]==ids.shape[1]+2
    assert model.last_report['visual_input_tokens']==16
    assert model.last_report['visual_llm_tokens']==4
    with pytest.raises(MemoryBudgetError):
        model.encode_images(torch.randn(24,patch),torch.tensor([[1,4,6]]))
    with pytest.raises(ValueError,match='Video'):
        model.generate(ids,pixel_values=pixels,image_grid_thw=grid,pixel_values_videos=pixels,max_new_tokens=1)


def test_qwen3vl_rejects_unknown_vision_weights(make_checkpoint):
    from safetensors.torch import save_file
    path,_,w=make_checkpoint('qwen3_vl')
    w['model.visual.surprise.weight']=torch.ones(1)
    save_file(w,str(path/'model.safetensors'))
    with pytest.raises(ValueError,match='Unknown tensor'):
        load_bounded_model(path,device='cpu',dtype=torch.float32)


def test_budget_inference_prepares_structured_local_image_without_cuda(make_checkpoint, tmp_path):
    from ollm.bounded.api import BudgetInference
    path,raw,_=make_checkpoint('qwen3_vl')
    budget=MemoryBudget(max_visual_tokens=16,max_images=1,max_output_tokens=2)
    image=tmp_path/'input.png'; image.write_bytes(b'fixture-path-only')
    class Tok:
        eos_token_id=2
        def get_vocab(self): return {'<|im_end|>':2}
    class Processor:
        tokenizer=Tok()
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]['content'][0]['image']==str(image.resolve())
            assert kwargs['return_dict'] and kwargs['return_tensors']=='pt'
            ids=torch.tensor([[1,raw['vision_start_token_id'],*([raw['image_token_id']]*4),raw['vision_end_token_id'],4]])
            v=raw['vision_config']; width=v['in_channels']*v['temporal_patch_size']*v['patch_size']**2
            return {'input_ids':ids,'attention_mask':torch.ones_like(ids),
                    'pixel_values':torch.zeros(16,width),'image_grid_thw':torch.tensor([[1,4,4]])}
    o=BudgetInference(path,model_key='qwen3-vl-2b-instruct',device='cpu',dtype=torch.float32,
                      budget=budget,load_tokenizer=False,allow_unqualified=True)
    o.processor=Processor(); o.tokenizer=o.processor.tokenizer
    ids,stops,extra=o.prepare_model_inputs([
        {'role':'user','content':[{'type':'image','image':str(image)},{'type':'text','text':'describe'}]}
    ])
    assert ids.shape[0]==1 and stops==[2]
    assert set(extra)=={'pixel_values','image_grid_thw'}
