import shutil

def cat_files(infiles, outfile, buffer=1024):
    """
    infiles: a list of files
    outfile: the file that will be created
    buffer: buffer size in bytes
    """
    with open(outfile, 'w+b') as tgt:
        for infile in infiles:
            with open(infile, 'r+b') as src:
                while True:
                    data = src.read(buffer)
                    if data:
                        tgt.write(data)
                    else:
                        break
cat_files(["./dsetsbyparts/OPTICAL_part."+str(a) for a in range(2)], "OPTICAL.mat")
shutil.rmtree('./dsetsbyparts')
