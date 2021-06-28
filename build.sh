set -eu

# Config vars
SRCDIR=$(pwd)/notebook
TGTDIR=$(pwd)/docs

for article_type in article; do
    for nb_path in $(ls -1 ${SRCDIR}/${article_type}/*.ipynb); do
        base=$(basename ${nb_path} .ipynb)

        outdir=${TGTDIR}/${article_type}/${base}
        if [ ! -e "${outdir}" ]; then
            echo "Create output directory at ${outdir}"
            mkdir -p ${outdir}
        fi

        echo jupyter nbconvert ${nb_path} --to markdown --output ${outdir}/index.md
        jupyter nbconvert ${nb_path} --to markdown --output ${outdir}/index.md
    done
done
