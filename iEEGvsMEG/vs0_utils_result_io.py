"""One canonical representation of nested results: tables, arrays, and an index."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def save_result(folder, result):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    arrays={}
    def encode(value, name='result'):
        if isinstance(value,pd.DataFrame):
            path=name+'.csv';value.to_csv(folder/path,index=False)
            return {'type':'table','path':path,'dtypes':{k:str(v) for k,v in value.dtypes.items()},'index':encode(value.index.tolist(),name+'_index'),'index_names':value.index.names}
        if isinstance(value,np.ndarray):
            if value.dtype.hasobject:
                return {'type':'object_array','items':encode(value.tolist(),name),'shape':list(value.shape)}
            arrays[name]=value
            return {'type':'array','key':name}
        if isinstance(value,dict):return {'type':'dict','items':[[encode(k,name+'_key'+str(i)),encode(v,(str(k) if name=='result' else name+'_'+str(k)) if isinstance(k,(str,int)) else name+'_'+str(i))] for i,(k,v) in enumerate(value.items())]}
        if isinstance(value,(tuple,list)):return {'type':'tuple' if isinstance(value,tuple) else 'list','items':[encode(v,name+'_'+str(i)) for i,v in enumerate(value)]}
        if isinstance(value,np.generic):value=value.item()
        if isinstance(value,Path):value=str(value)
        if value is None or isinstance(value,(str,int,float,bool)):return {'type':'scalar','value':value}
        raise TypeError(f'Unsupported result value {type(value)} at {name}')
    index=encode(result)
    # Object arrays are encoded as plain lists to avoid loading pickles.
    for k,v in list(arrays.items()):
        if v.dtype.hasobject:raise TypeError(f'Object array {k}: convert it to an explicit table/list.')
    np.savez_compressed(folder/'arrays.npz',**arrays)
    (folder/'result_index.json').write_text(json.dumps(index,indent=2))


def load_result(folder):
    folder=Path(folder)
    with np.load(folder/'arrays.npz',allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
    def decode(item):
        kind=item['type']
        if kind=='scalar':return item['value']
        if kind=='array':return arrays[item['key']]
        if kind=='object_array':return np.asarray(decode(item['items']),dtype=object).reshape(item['shape'])
        if kind=='dict':return {decode(k):decode(v) for k,v in item['items']}
        if kind in ('list','tuple'):
            values=[decode(v) for v in item['items']]
            return tuple(values) if kind=='tuple' else values
        if kind=='table':
            dtype={k:v for k,v in item['dtypes'].items() if v in ('object','string')}
            try:frame=pd.read_csv(folder/item['path'],dtype=dtype)
            except pd.errors.EmptyDataError:return pd.DataFrame(columns=list(item['dtypes']))
            index=decode(item['index'])
            if len(item['index_names'])>1:frame.index=pd.MultiIndex.from_tuples(index,names=item['index_names'])
            else:frame.index=pd.Index(index,name=item['index_names'][0])
            return frame
        raise ValueError(kind)
    return decode(json.loads((folder/'result_index.json').read_text()))
