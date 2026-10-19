import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch
from humanoidverse import train


class CheckpointCompletionTests(unittest.TestCase):
    def test_failed_save_keeps_old_checkpoint_and_success_replaces_all_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / train.CHECKPOINT_DIR_NAME
            checkpoint.mkdir()
            (checkpoint / 'model.version').write_text('old')
            (checkpoint / 'train_status.json').write_text('{"time": 100}')
            workspace = NS(work_dir=root, cfg=NS(checkpoint_buffer=True),
                agent=NS(save=lambda destination: (Path(destination) / 'model.version').write_text('new')))
            def buffer_save(destination):
                destination.mkdir(parents=True)
                (destination / 'buffer.version').write_text('new')
            replay = {'train': NS(save=buffer_save)}
            with patch.object(replay['train'], 'save', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    train.Workspace.save(workspace, 200, replay)
            self.assertEqual((checkpoint / 'model.version').read_text(), 'old')
            self.assertEqual(json.loads((checkpoint / 'train_status.json').read_text())['time'], 100)
            original_replace = Path.replace
            def fail_publish(path, destination):
                if path.name.startswith('.checkpoint-'):
                    raise OSError('publish failed')
                return original_replace(path, destination)
            with patch.object(Path, 'replace', fail_publish):
                with self.assertRaises(OSError):
                    train.Workspace.save(workspace, 200, replay)
            self.assertEqual((checkpoint / 'model.version').read_text(), 'old')
            # Simulate interruption between the two renames; resume must resolve the backup.
            checkpoint.replace(root / 'checkpoint.previous')
            self.assertEqual(json.loads((train.checkpoint_path(root) / 'train_status.json').read_text())['time'], 100)
            train.Workspace.save(workspace, 200, replay)
            self.assertEqual((checkpoint / 'model.version').read_text(), 'new')
            self.assertEqual((checkpoint / 'buffers/train/buffer.version').read_text(), 'new')
            self.assertEqual(json.loads((checkpoint / 'train_status.json').read_text())['time'], 200)
            self.assertFalse((root / 'checkpoint.previous').exists())
            self.assertEqual(list(root.glob('.checkpoint-*')), [])

    def test_training_budget_final_checkpoint_and_final_evaluation(self):
        for budget, interval, parallel, start in [(13,10,1,0), (5,10,1,0), (20,10,1,0), (12,8,4,0), (20,10,1,20)]:
            with self.subTest(budget=budget, parallel=parallel, start=start), tempfile.TemporaryDirectory() as folder:
                cfg = train.default_train_config().model_copy(update={
                    'num_env_steps': budget, 'checkpoint_every_steps': interval, 'eval_every_steps': interval,
                    'online_parallel_envs': parallel, 'buffer_size': 32, 'use_trajectory_buffer': False,
                    'num_seed_steps': 100, 'buffer_device': 'cpu', 'disable_tqdm': True, 'prioritization': False})
                steps, saves, evaluations = [], [], []
                def obs():
                    return {'state': np.full((parallel,1), len(steps), dtype=np.float32),
                            'time': np.full((parallel,1), len(steps), dtype=np.float32)}
                def step(action):
                    steps.append(action)
                    return obs(), np.zeros(parallel), np.zeros(parallel,dtype=bool), np.zeros(parallel,dtype=bool), {}
                env = NS(reset=lambda: (obs(),{}), step=step, close=lambda:None,
                         action_space=NS(sample=lambda: np.zeros((parallel,1),dtype=np.float32)))
                workspace = NS(training_with_expert_data=False, cfg=cfg, train_env=env, train_env_info={},
                    work_dir=Path(folder), _checkpoint_time=start, evaluations={}, evaluate=True,
                    eval=lambda t, **kw: evaluations.append(t),
                    agent=NS(device='cpu', maybe_update_rollout_context=lambda **kw: torch.zeros(parallel,1)),
                    save=lambda t, buffers: saves.append((t, start + len(steps)*parallel)))
                train.Workspace.train_online(workspace)
                self.assertEqual(start + len(steps)*parallel, budget)
                if budget > start:
                    self.assertEqual(saves[-1], (budget,budget))
                    self.assertEqual(len({t for t,_ in saves}), len(saves))
                    if budget % interval == 0:
                        self.assertEqual(evaluations[-1], budget)
                else:
                    self.assertEqual(saves, [])
                    self.assertEqual(evaluations, [])
