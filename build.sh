set -eu

# Config vars
SRCDIR=$(pwd)/notebook
TGTDIR=$(pwd)/docs/article

for nb_path in $(ls -1 ${SRCDIR}/*.ipynb); do
    base=$(basename ${nb_path} .ipynb)

    outdir=${TGTDIR}/${base}
    if [ ! -e "${outdir}" ]; then
	echo "Create output directory at ${outdir}"
	mkdir -p ${outdir}
    fi

    echo jupyter nbconvert ${nb_path} --to markdown --output ${outdir}/index.md
    jupyter nbconvert ${nb_path} --to markdown --output ${outdir}/index.md
done
