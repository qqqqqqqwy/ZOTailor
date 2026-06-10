#!/bin/sh
#

# Basedir on device
basedir=/data/local/tmp/llama.cpp

cli_opts=

branch=.
[ "$B" != "" ] && branch=$B

adbserial=
[ "$S" != "" ] && adbserial="-s $S"

adbhost=
[ "$H" != "" ] && adbhost="-H $H"

device="HTP0"
[ "$D" != "" ] && device="$D"

verbose=
[ "$V" != "" ] && verbose="GGML_HEXAGON_VERBOSE=$V"

sched=
[ "$SCHED" != "" ] && sched="GGML_SCHED_DEBUG=2" cli_opts="$cli_opts -v"

profile=
[ "$PROF" != "" ] && profile="GGML_HEXAGON_PROFILE=$PROF GGML_HEXAGON_OPSYNC=1"

opmask=
[ "$OPMASK" != "" ] && opmask="GGML_HEXAGON_OPMASK=$OPMASK"

nhvx=
[ "$NHVX" != "" ] && nhvx="GGML_HEXAGON_NHVX=$NHVX"

hmx=
[ "$HMX" != "" ] && hmx="GGML_HEXAGON_USE_HMX=$HMX"

ndev=
[ "$NDEV" != "" ] && ndev="GGML_HEXAGON_NDEV=$NDEV"

hb=
[ "$HB" != "" ] && hb="GGML_HEXAGON_HOSTBUF=$HB"

weight_layout=
[ "$WEIGHT_LAYOUT" != "" ] && weight_layout="GGML_HEXAGON_WEIGHT_LAYOUT=$WEIGHT_LAYOUT"

fused_lora=
[ "$FUSED_LORA" != "" ] && fused_lora="GGML_HEXAGON_FUSED_LORA=$FUSED_LORA"

split_trace=
[ "$SPLIT_TRACE" != "" ] && split_trace="GGML_HEXAGON_SPLIT_TRACE=$SPLIT_TRACE"

prompt_fa=
[ "$PROMPT_FA" != "" ] && prompt_fa="GGML_HEXAGON_FLASH_ATTN_PROMPT=$PROMPT_FA"

lora_debug=
[ "$LORA_DEBUG" != "" ] && lora_debug="GGML_HEXAGON_LORA_DEBUG=$LORA_DEBUG"

lora_debug_deep=
[ "$LORA_DEBUG_DEEP" != "" ] && lora_debug_deep="GGML_HEXAGON_LORA_DEBUG_DEEP=$LORA_DEBUG_DEEP"

lora_fast=
[ "$LORA_FAST" != "" ] && lora_fast="GGML_HEXAGON_LORA_FAST=$LORA_FAST"

lora_profile=
[ "$LORA_PROFILE" != "" ] && lora_profile="GGML_HEXAGON_LORA_PROFILE=$LORA_PROFILE"

rpcmem_uncached=
[ "$RPCMEM_UNCACHED" != "" ] && rpcmem_uncached="GGML_HEXAGON_RPCMEM_UNCACHED=$RPCMEM_UNCACHED"
[ "$GGML_HEXAGON_RPCMEM_UNCACHED" != "" ] && rpcmem_uncached="GGML_HEXAGON_RPCMEM_UNCACHED=$GGML_HEXAGON_RPCMEM_UNCACHED"

set -x

tool=$1; shift

adb $adbserial $adbhost shell " \
  cd $basedir; ulimit -c unlimited;        \
    LD_LIBRARY_PATH=$basedir/$branch/lib   \
    ADSP_LIBRARY_PATH=$basedir/$branch/lib \
    $verbose $sched $opmask $profile $nhvx $hmx $ndev $hb $weight_layout $fused_lora $split_trace $prompt_fa $lora_debug $lora_debug_deep $lora_fast $lora_profile $rpcmem_uncached ./$branch/bin/$tool $@ \
"
