set -eu

# Config vars
SRCDIR=notebook
TGTDIR=docs

for article_type in article blog; do
    for nb_path in $(ls -1 ${SRCDIR}/${article_type}); do
        base=$(basename ${nb_path})
        indir=${SRCDIR}/${article_type}/${base}

        outdir=${TGTDIR}/${article_type}/${base}
        if [ ! -e "${outdir}" ]; then
            echo "Create output directory at ${outdir}"
            mkdir -p ${outdir}
        fi

        # Convert Jupyter notebook to markdown
        echo jupyter nbconvert ${indir}/index.ipynb --to markdown --stdout
        jupyter nbconvert ${indir}/index.ipynb --to markdown --stdout >${outdir}/index.md

        # Copy images
        for png in $(ls -1 ${indir}/*.png ${indir}/*.jpg); do
            echo cp ${png} ${outdir}/
            cp ${png} ${outdir}/
        done
    done
done
