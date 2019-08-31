from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
import imageio, time, argparse, torch
from pathlib import Path

IPATH = Path('/Users/Daynan/ds/astrophysics/wilson/howard.astro.ucla.edu/pub/obs/drawings')
OPATH = Path('/Users/Daynan/ds/astrophysics/wilson/data/arrs')

parser = argparse.ArgumentParser()
parser.add_argument('-c','--num_cores', default=1, type=int)
args = parser.parse_args()

def load_filenames(inpath=IPATH, step=1):
    return np.asarray(sorted(list(inpath.glob('**/*.jpg'))[::step]))

class ImgMap:
    '''
    Asynchronous method for image I/O.
    Reads image, converts to numpy array, inverts
    Return tuple (file name, img array)
    '''
    def __init__(self,opath=OPATH, invert=True):
        self._invert=invert
        self.opath = opath

    def __call__(self,fname):
        if self._invert:
            img = np.invert(imageio.imread(fname))
        else:
            img = imageio.imread(fname)
        year = fname.parent.name
        if year == 'drawings': year = '2019'
        opath = Path(self.opath, year, fname.stem)
        torch.save(torch.ByteTensor(img), f'{opath}.pt')
        return True

def main(n):
    fnames = load_filenames()
    years = [fname.parent.name if fname.parent.name != 'drawings' else '2019' for fname in fnames]
    _ = [Path(OPATH, year).mkdir(parents=True, exist_ok=True) for year in years]

    proc = ImgMap()

    t0 = time.time()
    with ThreadPoolExecutor(n) as executor:
        fn = partial(proc)
        out = list(executor.map(fn, fnames))

    print('workers: {}\timg2arr runtime: {}'.format(n,time.time() - t0))
    print('imgs processed: {}'.format(sum(out)))

if __name__ == "__main__":
    main(n = args.num_cores)