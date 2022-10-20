from argparse import ArgumentParser
from collections import defaultdict
import glob
import json
import os
import re
from typing import List, NamedTuple

from natsort import natsorted
from polyaxon.tracking import Run
import yaml


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--param-file', action='append', default=[])
    parser.add_argument('--metric-file', action='append', default=[])
    parser.add_argument('--data-file', action='append', default=[])
    parser.add_argument('--tag', action='append', default=[])
    parser.add_argument('--capture-png', action='store_true')
    
    return parser.parse_args()


def startswith(s: str, vals):
    return any(s.startswith(v) for v in vals)


def parse_gin_line(line):
    lhs, rhs = line.split('=')
    lhs = lhs.strip().replace('.', '__').replace('/', '__')
    rhs = rhs.strip()
    if startswith(rhs, ['@', '%', '"', "'"]):
        rhs = rhs.replace('"', '').replace("'", '')
    elif startswith(rhs, ['\\', '[', '(']):
        pass
    elif rhs in ['True', 'False']:
        rhs = rhs == 'True'
    elif '.' in rhs:
        rhs = float(rhs)
    elif rhs == 'None':
        rhs = None
    else:
        rhs = int(rhs)
    return lhs, rhs


def load_gin(stream):
    return {
        parse_gin_line(line)[0]: parse_gin_line(line)[1]
        for line in stream
        if '=' in line and not line.startswith('#')
    }


def get_loader(fname):
    if fname.lower().endswith(".json"):
        return json.load
    if fname.lower().endswith(".yml") or fname.lower().endswith(".yaml"):
        return yaml.load
    if fname.lower().endswith(".gin"):
        return load_gin
    raise ValueError(f'Unsupported file format: {fname}')


def load(fname):
    loader = get_loader(fname)
    with open(fname) as infile:
        return loader(infile)


def load_values(fnames):
    vals = {}
    for fname in fnames:
        vals.update(**load(fname))
    return vals


def load_datasets(fnames):
    from divik._cli._data_io import load_data
    return (
        {'content': load_data(fname), 'name': fname, 'path': fname}
        for fname in fnames
    )


SerialImages = NamedTuple('SerialImages', [
    ('paths', List[str]),
    ('name', str)
])


def discover_png(dirname: str):
    pngs = glob.glob(os.path.join(dirname, '*.png'))
    fname_pattern = re.compile('.+[_-][0-9]+.[pP][nN][gG]$')
    suffix_pattern = re.compile('[_-][0-9]+.[pP][nN][gG]$')

    standalones = [p for p in pngs if fname_pattern.fullmatch(p) is None]
    
    grouped = [p for p in pngs if p not in standalones]
    serial = defaultdict(list)
    for p in natsorted(grouped):
        name = os.path.split(suffix_pattern.sub('', p))[1]
        serial[name].append(p)
    grouped = [
        SerialImages(paths=paths, name=name)
        for name, paths in serial.items()
    ]
    
    return standalones + grouped


def main():
    args = parse_args()
    experiment = Run()
    params = load_values(args.param_file)
    if params:
        experiment.log_inputs(**params)
    metrics = load_values(args.metric_file)
    if metrics:
        experiment.log_metrics(**metrics)
    if args.tag:
        experiment.log_tags(args.tag)
    for dataset in load_datasets(args.data_file):
        experiment.log_data_ref(**dataset)
    if args.capture_png:
        imgs = discover_png(experiment.get_outputs_path())
        for img in imgs:
            if isinstance(img, str):
                experiment.log_image(img)
            elif isinstance(img, SerialImages):
                for idx, path in enumerate(img.paths):
                    experiment.log_image(path, name=img.name, step=idx)
            else:
                raise NotImplementedError('We should never get here.')


if __name__ == '__main__':
    main()