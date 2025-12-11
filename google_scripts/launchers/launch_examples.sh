#!/bin/bash
xmanager launch google_scripts/launchers/launch_video_r1_test.py -- \
  --xm_resource_alloc=xcloud/xcloud-shared-user \
  --exp_name=FT_baseline_exps \
  --platform=h100=8 \
  --interactive_only=True

xmanager launch google_scripts/launchers/launch_video_r1_test.py -- \
  --xm_resource_alloc=xcloud/xcloud-shared-user \
  --exp_name=video_r1_sat_zebra_latentabltation \
  --platform=h100=8 \
  --interactive_only=True

xmanager launch google_scripts/launchers/launch_video_r1_test.py -- \
  --xm_resource_alloc=xcloud/xcloud-shared-user \
  --exp_name=sims_qwen \
  --platform=h100=6 \
  --interactive_only=True

xmanager launch google_scripts/launchers/launch_video_r1_test.py -- \
  --xm_resource_alloc=xcloud/xcloud-shared-user \
  --exp_name=video_r1_zebra_ftfrozen \
  --platform=h100=6 \
  --interactive_only=True

xmanager launch google_scripts/launchers/launch_video_r1_test.py -- \
  --xm_resource_alloc=xcloud/xcloud-shared-user \
  --exp_name=video_r1_zebra_grpo \
  --platform=h100=6 \
  --interactive_only=True

xmanager launch google_scripts/launchers/launch_video_r1_test.py -- \
  --xm_resource_alloc=xcloud/xcloud-shared-user \
  --exp_name=zebracot_mirage \
  --platform=h100=5 \
  --interactive_only=True


# run_sft_qwen_SAT_VidR1
xmanager launch google_scripts/launchers/launch_video_r1_test.py -- \
  --xm_resource_alloc=xcloud/xcloud-shared-user \
  --exp_name=run_sft_qwen_SAT_VidR1_pause \
  --platform=a100_80gib=5

# run_sft_qwen_SAT_VidR1_query
xmanager launch google_scripts/launchers/launch_video_r1_test.py -- \
  --xm_resource_alloc=xcloud/xcloud-shared-user \
  --exp_name=run_sft_qwen_SAT_VidR1_query \
  --platform=a100_80gib=5x`