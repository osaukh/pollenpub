'''Copyright (C) 2019 Erik Swiss Federal Institute of Technology (ETH Zurich), Matthias Meyer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Changes to original version: Adopted to work with multi-layer pollen data
'''

import os
import pandas as pd
import argparse
import numpy as np 
from os.path import join 
from pathlib import Path    

'''
Script generates a training config based on a directory containing two folders (*labels* and *layers*) 
and one text file containing the classes (*classes.names*).
'''

parser = argparse.ArgumentParser()
parser.add_argument("--dir",'-f', type=str, default="data/pollen_deployment/test.txt", help="path to model definition file")
parser.add_argument("--name",'-n', type=str, default="pollen", help="base name of the dataset")
parser.add_argument("--output_dir",'-o', type=str, default="data/pollen_joint/", help="where to store the fold information")
parser.add_argument("--merge_set",'-m', type=str, default=None, help="optional set to be merged with training set")
parser.add_argument('-K', type=int, default=1, help="Number of folds to generate")
parser.add_argument("--stats_only", action='store_true',  help="Output dataset statistics only")
args = parser.parse_args()

# df = pd.read_csv(args.filename, sep=" ", header=None, names=['filename'])
p = Path(args.dir)

filenames = {'filename':[path.relative_to(p) for path in p.glob('layers/*')]}
df = pd.DataFrame(filenames)

df.sort_values('filename',inplace=True)
df['group'] = df['filename'].apply(lambda x: str(x).split('/')[-1].split('-')[0])

groups = df['group'].unique()

print("Number of layers \t", len(df))
print("Number of stacks \t", len(groups))
print("Average Number Layers per Stack \t", len(df)/len(groups))

if args.stats_only:
    total_num_pollen = 0
    layers_with_pollen = 0
    layers_pollen = []
    for filename in df['filename']:
        label_file = p/Path(str(filename).replace("layers", "labels").replace(".png", ".txt").replace(".jpg", ".txt"))
        if not label_file.exists():
            layers_pollen += [0]
            continue
    # for label_file in p.glob('labels/*.txt'):
        num_pollen = 0
        with open(label_file) as f:
            for line in f: # count the number of lines == number of labeled pollen
                num_pollen += 1
        if num_pollen > 0:
            layers_with_pollen += 1
        
        layers_pollen += [num_pollen]
        total_num_pollen += num_pollen
    print('Total number of labeled pollen',total_num_pollen)
    print('Total number of layers with pollen',layers_with_pollen)

    df['num_pollen'] = layers_pollen

print(f"Average number of stacks per train \t {len(groups)/args.K * (args.K-1)}")
print(f"Average number of stacks per test \t {len(groups)/args.K}")

if args.merge_set:
    merge_df = pd.read_csv(args.merge_set, sep="\n", header=None, names=['filename'])
    print(f'Loaded merge set with {len(merge_df)} items')
else:
    merge_df = None

folds = []
for k in range(args.K):
    groups_fold = groups[k::args.K]
    fold_df = df[df['group'].apply(lambda x: x in groups_fold)]
    folds.append(fold_df)
    print(f"Fold {k}: Number of stacks {len(groups_fold)}")
    print(f"Fold {k}: Number of pollen {fold_df['num_pollen'].sum()}")
    print(f"Fold {k}: Layers with pollen {(fold_df['num_pollen']>0).sum()}")

# create training set
indices = np.arange(args.K).astype(np.int)
for k in range(args.K):
    if args.K == 1:
        base = ''
    else:
        base = f"fold{k}_"
    train_filename = join(args.output_dir,f'{base}train.txt')
    test_filename = join(args.output_dir,f'{base}test.txt')

    test_set = folds[k]
    print(f"Fold{k}: Images in test set \t {len(test_set)}")

    if not args.stats_only:
        test_set.to_csv(test_filename,columns=['filename'],index=False,header=False, sep='\n')

    if args.K > 1:
        train_folds = [folds[i] for i in indices[indices!=k]]
        train_set = pd.concat(train_folds)
        if merge_df is not None:
            train_set = train_set.append(merge_df)
    
        print(f"Fold{k}: Images in train set \t {len(train_set)}")
        if not args.stats_only:
            train_set.to_csv(train_filename,columns=['filename'],index=False,header=False, sep='\n')

    if not args.stats_only:
        with open(f"config/{args.name}{base[:-1]}.data",'w') as f:
            f.write('classes= 1\n')
            f.write('train='+train_filename + '\n')
            f.write('valid='+test_filename + '\n')
            f.write('base_dir='+join(args.output_dir,'') + '\n')
            f.write('names='+join(args.output_dir,'classes.names') + '\n')


