#!/bin/bash

source ~/miniconda3/bin/activate pyPSS-env

#set -x

sh_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

src_dir="$(realpath "${sh_dir}/src")"

data_dir="$(realpath "${sh_dir}/data")"

if (( $# != 3 )); then
    echo "Usage: run_model.s model_name beta run"
    exit 1
fi

model=$1
beta=$(python3 -c "print('{:.1f}'.format($2))")
run=$(python3 -c "print('{:d}'.format($3))")

tauc=1.0
K=1 # 0.1
alpha=0.1
chi=0.0
gamma=5.0
lambd=0.5
mu=1
B=1 # 1
D=2
a=1 # 0.1
b=1 # 0.1
ct=0.1
epsilont=0.5

T=40
n_steps=4e+4
dt_dump=1.0
mx=100
my=100
dx=0.5
dy=0.5

save_dir="${sh_dir}/data/$model/alpha_${alpha}_beta_${beta}_chi_${chi}_gamma_${gamma}/run_${run}"

if [ ! -d $save_dir ]; then
    mkdir -p $save_dir
fi

params_file="${save_dir}/parameters.json"

echo \
"
{
    "\"run\"" : $run,
    "\"T\"" : $T,
    "\"n_steps\"" : $n_steps,
    "\"dt_dump\"" : $dt_dump,
    "\"K\"" : $K,
    "\"tauc\"" : $tauc,
    "\"alpha\"" : $alpha,
    "\"beta\"" : $beta,
    "\"gamma\"" : $gamma,
    "\"lambd\"" : $lambd,
    "\"mu\"" : $mu,
    "\"chi\"": $chi,
    "\"B\"": $B,
    "\"D\"": $D,
    "\"a\"": $a,
    "\"b\"": $b,
    "\"ct\"" : $ct,
    "\"epsilont\"" : $epsilont,
    "\"mx\"" : $mx,
    "\"my\"" : $my,
    "\"dx\"" : $dx,
    "\"dy\"" : $dy
}
" > $params_file

python3 -m models.$model -s $save_dir

#python3 -m src.analysis.create_avgs -s $save_dir

#python3 -m src.analysis.create_videos_rho -s $save_dir