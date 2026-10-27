#!/bin/bash
# re-render every scene where the robot carries something, with the payload-trained wheel + walk policies
S=/tmp/claude-1000/-home-ubuntu/16951df8-1513-4784-9767-fdb283224674/scratchpad
OUT=$HOME/dobot_tools/resort/results_rtx
cd ~/IsaacLab
R=logs/rsl_rl/dobot_quad_wheel_rough_payload/2026-10-23_10-01-26/model_3799.pt
WALK=logs/rsl_rl/dobot_quad_wheel_walk_payload/2026-10-23_11-52-48/exported/policy.pt
X="uv run --extra ovrtx --extra video python"
mkdir -p $OUT/s6; mv $OUT/s6 $OUT/s6_nopayload_bak 2>/dev/null; mkdir -p $OUT/s6
$X ~/dobot_tools/resort/patrol.py --rtx 1 --scenario 6 --out $OUT/s6 --walk_policy $WALK --max_minutes 6 --checkpoint $R --num_envs 1 --viz newton_rtx > $S/pl_s6.log 2>&1
echo "== s6 $?"; grep -E "EVENT|Traceback" $S/pl_s6.log | cut -c1-110
for job in breakfast guide; do
  mv $OUT/$job $OUT/${job}_nopayload_bak 2>/dev/null; mkdir -p $OUT/$job
  $X ~/dobot_tools/resort/services.py --rtx 1 --job $job --out $OUT/$job --walk_policy $WALK --checkpoint $R --num_envs 1 --viz newton_rtx > $S/pl_$job.log 2>&1
  echo "== $job $?"; grep -E "RESULT\]|Traceback" $S/pl_$job.log | cut -c1-200
done
D=logs/rsl_rl/dobot_quad_wheel_dance/2026-10-23_10-52-01/model_1499.pt
U=logs/rsl_rl/dobot_quad_wheel_rearup/2026-10-23_11-18-21/model_1999.pt
mv $OUT/party $OUT/party_nopayload_bak 2>/dev/null; mkdir -p $OUT/party
$X ~/dobot_tools/resort/party.py --out $OUT/party --rtx 1 --dance_policy $D --rearup_policy $U --checkpoint $R --num_envs 1 --viz newton_rtx > $S/pl_party.log 2>&1
echo "== party $?"; grep -E "RESULT\]|Traceback" $S/pl_party.log | cut -c1-200
echo ALL DONE
