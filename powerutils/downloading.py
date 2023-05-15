from ftplib import FTP
from pathlib import Path, PurePosixPath

from powerutils.common import new_dir

class FtpDownloader:
    '''
    FTP下载器.

    下载文件和目录的方法不会检查路径是否合法,
    因为FTP协议要实现这个还有点麻烦. 会由ftplib报错.
    '''
    def __init__(
        self, host, port, user, passwd,
        timeout=None, encoding='utf-8'
    ):
        self.ftp = FTP(timeout=timeout, encoding=encoding)
        self.ftp.connect(host=host, port=port)
        self.ftp.login(user=user, passwd=passwd)

    def download_file(self, filepath_remote, filepath_local):
        '''下载一个文件到本地.'''
        filepath_remote = PurePosixPath(filepath_remote)
        filepath_local = Path(filepath_local)
        with open(str(filepath_local), 'wb') as f:
            self.ftp.retrbinary(f'RETR {str(filepath_remote)}', f.write)
        print(f'[OK] {filepath_remote.name}')

    def download_directory(
        self, dirpath_remote, dirpath_local,
        filter=None, overwrite=False, keep_tree=False,
    ):
        '''
        递归下载一个目录到本地.

        过滤掉路径不满足filter的文件.
        overwrite控制是否覆盖本地已有的文件.
        keep_tree指定是否保留远程目录的结构.
        '''
        dirpath_remote = PurePosixPath(dirpath_remote)
        print(f'> {str(dirpath_remote)}')
        dirpath_local = Path(dirpath_local)
        new_dir(dirpath_local)

        # 通过dir返回的文件属性区分文件和目录.
        lines = []
        self.ftp.dir(str(dirpath_remote), lines.append)
        for line in lines:
            parts = line.split()
            attrs, name = parts[0], parts[-1]
            path = dirpath_remote / name
            if attrs[0] == '-':
                if filter is None or filter(path):
                    filepath_local = dirpath_local / path.name
                    if overwrite or not filepath_local.exists():
                        self.download_file(path, filepath_local)
            elif attrs[0] == 'd':
                if keep_tree:
                    dirpath_temp = dirpath_local / path.name
                else:
                    dirpath_temp = dirpath_local
                self.download_directory(
                    path, dirpath_temp,
                    filter, overwrite, keep_tree
                )
            else:
                continue

    def close(self):
        self.ftp.quit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()