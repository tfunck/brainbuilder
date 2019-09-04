#!/bin/bash
if [[ ! -f "base.simg" ]] ; then
    singularity build base.simg base/Singularity 
fi

if [[ ! -f "ants.simg" ]]; then
    singularity build ants.simg ants/Singularity 
fi

if [[ ! -f jumni-receptor-atlas.simg ]]; then
    singularity build jumni-receptor-atlas.simg jumni-receptor-atlas/Singularity 
fi
