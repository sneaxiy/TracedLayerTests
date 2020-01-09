TAR_FILE="simple-examples.tgz"

wget http://www.fit.vutbr.cz/~imikolov/rnnlm/${TAR_FILE}

tar -xzvf ${TAR_FILE}

rm -rf ${TAR_FILE}
