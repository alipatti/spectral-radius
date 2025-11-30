set files paper/**/*.{tex,cls,sty,bib} figures/**/*.{pdf,pgf}
set tarball /Users/ali/Downloads/for-arxiv.tar.gz

# make tarball and list files
tar -cz -f $tarball $files
echo $tarball && tar -tf $tarball
