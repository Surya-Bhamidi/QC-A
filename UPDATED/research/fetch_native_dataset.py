"""Resume the official public native-224 dataset with four bounded ranges."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
from pathlib import Path
import shutil
import time
import urllib.request

URL='https://zenodo.org/records/10519652/files/pneumoniamnist_224.npz?download=1'
MD5='d6a3c71de1b945ea11211b03746c1fe1'
DEST=Path('data/native_224/pneumoniamnist.npz')


def download_range(start,end,path):
    expected=end-start+1
    if path.exists() and path.stat().st_size==expected:
        return path
    for attempt in range(3):
        try:
            request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(request,timeout=60) as response:
                assert response.status==206,'Server did not honor a bounded range'
                assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                with path.open('wb') as out:
                    shutil.copyfileobj(response,out,length=1024*1024)
            assert path.stat().st_size==expected
            print('Downloaded range',start,end,'bytes',expected,flush=True)
            return path
        except Exception:
            if attempt==2:
                raise
            time.sleep(attempt+1)


def main():
    # Every write is to a fixed file beneath the intended project data folder.
    root=Path.cwd().resolve();parent=(root/'data/native_224').resolve()
    assert parent.is_relative_to(root)
    dest=DEST.resolve();assert dest.parent==parent
    parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and hashlib.md5(dest.read_bytes()).hexdigest()==MD5:
        print('Reused verified official dataset',flush=True)
        return
    request=urllib.request.Request(URL,headers={'Range':'bytes=0-0'})
    with urllib.request.urlopen(request,timeout=60) as response:
        assert response.status==206,'Resume requires HTTP range support'
        total=int(response.headers['Content-Range'].split('/')[-1])
    prefix=dest.with_suffix('.npz.prefix')
    if dest.exists():
        assert not prefix.exists(),'Preserve the existing prefix before resuming'
        dest.replace(prefix)
    offset=prefix.stat().st_size if prefix.exists() else 0
    assert 0<=offset<total
    if offset:
        with prefix.open('rb') as source:
            assert source.read(2)==b'PK','Unexpected archive prefix'
    remaining=total-offset;width=(remaining+3)//4
    ranges=[(start,min(start+width-1,total-1),parent/f'pneumoniamnist.range-{i}.part')
            for i,start in enumerate(range(offset,total,width))]
    print('Resuming',offset,'of',total,'bytes with',len(ranges),'bounded requests',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(download_range,*entry) for entry in ranges]):
            future.result()
    assembled=dest.with_suffix('.npz.assembled')
    with assembled.open('wb') as out:
        if prefix.exists():
            with prefix.open('rb') as source:
                shutil.copyfileobj(source,out,length=1024*1024)
        for _,_,part in ranges:
            with part.open('rb') as source:
                shutil.copyfileobj(source,out,length=1024*1024)
    checksum=hashlib.md5(assembled.read_bytes()).hexdigest()
    assert checksum==MD5,(checksum,'Official dataset checksum mismatch')
    assembled.replace(dest)
    print('Verified official native-224 dataset',dest.stat().st_size,checksum,flush=True)


if __name__=='__main__':
    main()
