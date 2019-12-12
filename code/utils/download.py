"""MIT License

Copyright (c) 2019, Swiss Federal Institute of Technology (ETH Zurich), Matthias Meyer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import wget
from pathlib import Path
import os
import argparse
import zipfile


parser = argparse.ArgumentParser(description="Download Pollen Files")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default='../',
    help="The path to where the data should be downloaded",
)
parser.add_argument(
    "-f", "--files", type=str, nargs="+", help="The files to be downloaded",
)

args = parser.parse_args()

data_dir = Path(args.path)
print(f"Downloading files to {data_dir}")
os.makedirs(data_dir, exist_ok=True)

base_url = (
    "https://zenodo.org/record/3572653/files/"
)

if not args.files:
    # download all files
    print(
        "WARNING: Downloading all files, which are several GB of data. "
        "Make sure you are using an appropriate internet connection and have enough space left on your disk"
    )
    files = [
        "weights.zip",
        "data.zip",
    ]
else:
    files = args.files


for item in files:
    filename = data_dir / item
    if filename.exists():
        print(f"Not downloading {item}. It already exists.")
        continue
    print(f"Downloading {item}")
    try:
        wget.download(base_url + item, str(filename))
    except Exception as e:
        print(e)

print("Extracting files")
for item in files:
    filename = data_dir / item
    with zipfile.ZipFile(filename, "r") as zip_file:
        zip_file.extractall(data_dir)
