import re
import warnings

import pandas as pd
import transformers
from hfutils.operate import get_hf_client

from imgutils.preprocess.transformers.base import _FN_CREATORS

hf_client = get_hf_client()
df = pd.read_parquet(hf_client.hf_hub_download(
    repo_id='deepghs/hf_models_preprocessors',
    repo_type='dataset',
    filename='repos.parquet'
))
df = df[~df['image_processor_type'].isnull()]
df = df.sort_values(by=['likes'], ascending=[False])

d_repo_count = {
    item['image_processor_type']: item['count']
    for item in df['image_processor_type'].value_counts().reset_index().to_dict('records')
}

d_create_functions = {}
for xfn in _FN_CREATORS:
    xname = xfn.__name__
    matching = re.fullmatch('^create_transforms_from_(?P<name>[\s\S]+)_processor$', xname)
    if not matching:
        warnings.warn(f'Cannot determine transformer type of {xfn!r}.')
        continue
    raw_name = matching.group('name').replace('_', '').lower()
    d_create_functions[raw_name] = xname

suffix = 'ImageProcessor'

rows = []
for name in dir(transformers):
    if name.endswith(suffix) and isinstance(getattr(transformers, name), type) \
            and issubclass(getattr(transformers, name), transformers.BaseImageProcessor) \
            and getattr(transformers, name) is not transformers.BaseImageProcessor:
        cls = getattr(transformers, name)
        pname = name[:-len(suffix)].lower()

        rows.append({
            'Name': name,
            'Supported': '✅' if pname in d_create_functions else '❌',
            'Repos': d_repo_count.get(name, 0),
            'Function': f':func:`{d_create_functions[pname]}`' if pname in d_create_functions else 'N/A'
        })

df = pd.DataFrame(rows)
df['Ratio'] = (df['Repos'] / df['Repos'].sum()).map(lambda x: f'{x * 100.0:.2f}%')
df = df.sort_values(by=['Repos', 'Supported', 'Name'], ascending=[False, True, True])
df = df[['Name', 'Supported', 'Repos', 'Ratio', 'Function']]
df = df[df['Repos'] >= 5]
print(df.to_markdown(headers='keys', tablefmt='rst', index=False))
