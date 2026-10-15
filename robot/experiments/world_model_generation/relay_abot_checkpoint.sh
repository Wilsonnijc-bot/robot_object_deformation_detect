#!/bin/bash
# Relay the ABot-PhysWorld checkpoint ModelScope -> Mac -> ERIS in 512 MiB parts (ERIS cannot reach
# modelscope.cn; the Mac lacks space for the whole file). Parts already on ERIS at full size are skipped,
# so rerunning resumes. Stops at the first failure; never opens a new SSH login.
set -u
URL="https://www.modelscope.cn/models/amap_cvlab/Abot-PhysWorld/resolve/master/abotpw_i2v_480p.safetensors"
SIZE=32789894056
SHA=41bcf456602fb9e38a9c3fa9b1e7f7357068a2c29c351f441f9768a7a8e8a0c1
CHUNK=$((512 * 1024 * 1024))
R=/tmp/eris_ctl/r.sh
DEST=/scratch/z/zy992/models/ABot-PhysWorld
PARTS=$DEST/parts
$R "mkdir -p $PARTS" || exit 1
N=$(( (SIZE + CHUNK - 1) / CHUNK ))
existing=$($R "cd $PARTS && stat -c '%n %s' part.* 2>/dev/null" || true)
for ((i = 0; i < N; i++)); do
  start=$((i * CHUNK)); end=$((start + CHUNK - 1)); ((end >= SIZE)) && end=$((SIZE - 1))
  want=$((end - start + 1)); name=$(printf 'part.%04d' "$i")
  if grep -qx "$name $want" <<<"$existing"; then continue; fi
  if ! curl -sfL --retry 3 -r "$start-$end" "$URL" | $R "cat > $PARTS/$name.tmp && [ \$(stat -c %s $PARTS/$name.tmp) -eq $want ] && mv $PARTS/$name.tmp $PARTS/$name"; then
    echo "FAILED at $name"; exit 1
  fi
  echo "$(date +%T) $name/$N done"
done
$R "cd $DEST && cat parts/part.* > abotpw_i2v_480p.safetensors.tmp && echo '$SHA  abotpw_i2v_480p.safetensors.tmp' | sha256sum -c - && mv abotpw_i2v_480p.safetensors.tmp abotpw_i2v_480p.safetensors && rm -rf parts && ls -la" || { echo "ASSEMBLY/CHECKSUM FAILED"; exit 1; }
echo RELAY_DONE
