# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Launches a Jupyter Notebook on GCP.

xmanager launch scripts/launchers/launch_video_r1_test.py -- \
  --xm_resource_alloc=xcloud/xcloud-shared-user \
  --exp_name=video_r1_test \
  --platform=a100_80gib=5

xmanager gcp port-forward --job=$JOB --ports=8000:8888

For Cloudtop users:
- You can navigate to http://$USER.c.googlers.com:8000 from your laptop
to access the notebook.

For Workstation users:
- You can navigate to http://$USER.mtv.corp.google.com:8000 from your
laptop to access the notebook.

Warning: Navigating to http://$USER.mtv.corp.google.com:8000 may
not work due to reverse-proxy issues. In that case, you can

1) Navigate to go/crd and open the browser to http://localhost:8000.

2) As an alternative to using go/crd, you can install Jupyter websocket extensions:

pip install --upgrade jupyter_http_over_ws>=0.0.7 && \
    jupyter serverextension enable --py jupyter_http_over_ws

jupyter notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=9999 \
    --NotebookApp.port_retries=0

"""

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc
from xmanager.contrib.internal import requirements_flag
from google3.learning.deepmind.xmanager2.gcp.launch import flags as gcp_flags

_JOB_TIMEOUT = flags.DEFINE_string(
    'job_timeout',
    '600h',
    (
        'Up time in seconds before terminating. '
        'Be Googley! Running unnecessary idle jobs prevents other users from '
        'scheduling their jobs.'
    ),
)
# There are only CPU and GPU images because TPUs run in 2-VM mode.
# https://cloud.google.com/ai-platform/deep-learning-containers/docs/choosing-container
_BASE_IMAGE = flags.DEFINE_string(
    'base_image',
    # 'gcr.io/deeplearning-platform-release/pytorch-gpu.1-13',
    'gcr.io/deeplearning-platform-release/huggingface-pytorch-training-cu121.2-3.transformers.4-42.ubuntu2204.py310',
    'Base image that has Conda and jupyterlab installed.',
)
_PIP_PACKAGES = flags.DEFINE_list(
    'pip',
    (),
    'List of pip or conda packages to install onto the Jupyter kernel.',
)
_ACCELERATOR = requirements_flag.DEFINE_requirements(
    'platform',
    'a100_80gib=2',
    'Accelerator specification. Format: <GPU>=<count> or <TPU>=<topology>.',
    short_name='t',
)

_EXPNAME = flags.DEFINE_string(
    'exp_name',
    'VideoR1',
    'Experiment name.',
    short_name='exp',
)

_CKPT_PATH = flags.DEFINE_string(
    'ckpt_path',
    '',
    'Checkpoint path.',
    short_name='ckpt',
)

_INTERACTIVE_ONLY = flags.DEFINE_bool(
    'interactive_only',
    False,
    'if True, doesnt launch the training job, just sets up the environment.',
    short_name='i',
)

_LDAP = flags.DEFINE_string(
    'ldap',
    '',
    'ldap of the user.',
    short_name='ldap',
)


def main(argv) -> None:
  if len(argv) > 2:
    raise app.UsageError('Too many command-line arguments.')
  exp_name = _EXPNAME.value
  interactive_only = _INTERACTIVE_ONLY.value
  ldap = _LDAP.value

  ckpt_path = _CKPT_PATH.value
  ckpt_copy_cmd = ''
  if ckpt_path != '':
    ckpt_exp_name = ckpt_path.split('/')[-2]
    ckpt_copy_cmd = "mkdir /home/jupyter/checkpoints/"
    ckpt_copy_cmd += f" && mkdir /home/jupyter/checkpoints/{ckpt_exp_name}"
    ckpt_copy_cmd += f' && gsutil -m cp -r {ckpt_path} /home/jupyter/checkpoints/{ckpt_exp_name}/.'

  launch_cmd = ''
  if not interactive_only:
    launch_cmd = f'cd Video-R1 && bash google_scripts/launch_scripts/{exp_name}.sh'

  with xm_abc.create_experiment(
      experiment_title=exp_name,
  ) as experiment:
    executor = xm_abc.Gcp(
        requirements=xm.JobRequirements(**_ACCELERATOR.value),
    )
    # conda_packages = ' '.join(_PIP_PACKAGES.value)

    # read pip packages from requirements.txt
    with open('requirements/requirements.txt', 'r') as f:
      pip_packages = f.read().strip().splitlines()

    pip_packages = ' '.join(pip_packages)

    [server] = experiment.package([
        xm.python_container(
            base_image=_BASE_IMAGE.value,
            docker_instructions=[
                'RUN mkdir /workdir',
                'WORKDIR /workdir',
                'RUN apt-get update -y',
                'RUN cd /workdir',
                (
                    'RUN echo "deb'
                    ' [signed-by=/usr/share/keyrings/cloud.google.gpg]'
                    ' https://packages.cloud.google.com/apt cloud-sdk main" |'
                    ' tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&'
                    ' curl'
                    ' https://packages.cloud.google.com/apt/doc/apt-key.gpg |'
                    ' gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg &&'
                    ' apt-get update -y && apt-get install google-cloud-cli -y'
                ),
                'RUN python3 -m pip install --upgrade pip',
                'RUN pip uninstall -y psutil',
                'RUN pip install --force-reinstall psutil==5.9.3',
                'RUN pip install jupyterlab',
                'RUN apt-get install -y vim',
                'RUN apt-get install -y unzip',
                'RUN apt-get install -y tmux',
                'RUN apt-get install -y zstd',
                'WORKDIR /home/jupyter/',
                'RUN git clone https://github.com/arijitray1993/Video-R1.git',
                'RUN cd Video-R1/src/r1-v && pip install -e ".[dev]"',
                f'RUN python3 -m pip --no-cache-dir install {pip_packages}',
                'RUN cd Video-R1/src/qwen-vl-utils && pip install -e .[decord]',
                'RUN cd Video-R1/transformers-main && pip install .',
                'RUN pip install datasets==3.0.2',
                'RUN cd Video-R1 && git clone https://github.com/EvolvingLMMs-Lab/lmms-eval',
                #'RUN curl -LsSf https://astral.sh/uv/install.sh | sh',
                #'RUN uv venv lmms_eval && source lmms_eval/bin/activate',
                'RUN cd Video-R1/lmms-eval && pip install -e .',
                'RUN chown -R 1000:root /home',
                # 'RUN chown -R 1000:root /usr/local',
            ],
            entrypoint=xm.CommandList([
                'nohup jupyter lab --ip="*" --NotebookApp.token="" &',
                'gsutil -m rsync -x ".git/|transformers-main/" -r gs://xcloud-shared/arijitray/Video-R1 Video-R1',
                'source Video-R1/google_scripts/google_prep_scripts/setup_xcloud.sh',
                'cp Video-R1/google_scripts/google_prep_scripts/pull_code.sh .',
                ckpt_copy_cmd,
                launch_cmd,
                'sleep ' + _JOB_TIMEOUT.value,
            ]),
            executor_spec=executor.Spec(),
        ),
    ])
    experiment.add(xm.Job(executable=server, executor=executor))


if __name__ == '__main__':
  app.run(main)
