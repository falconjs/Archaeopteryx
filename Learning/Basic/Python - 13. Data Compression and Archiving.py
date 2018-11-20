"""
https://docs.python.org/3.6/library/zipfile.html


"""

# 13.5. zipfile — Work with ZIP archives
import zipfile

"""
class zipfile.ZipFile(file, mode='r', compression=ZIP_STORED, allowZip64=True)
Open a ZIP file, where file can be a path to a file (a string), a file-like 
object or a path-like object.
"""
filename = './Archaeopteryx/Learning/Basic/13DCZIP.zip'

with zipfile.ZipFile(filename) as f:
    # Return a list of archive members by name.
    print(f.namelist())
    # ['file_a.txt', 'file_b.txt', 'file_c.txt']
    print(f.read('file_a.txt'))
    # b'a\r\nA\r\n'
    with f.open('file_b.txt') as fb:
        print(fb.read())
        # b'b\r\nB\r\n'

f.close()

