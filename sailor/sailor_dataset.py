from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import h5py

class Sailor(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Default robot state, consists of [3x robot ee pos, '
                                '3x ee quat, 1x gripper state].',
                        ),
                        'state_ee': tfds.features.Tensor(
                            shape=(16,),
                            dtype=np.float32,
                            doc='End-effector state, represented as 4x4 homogeneous transformation matrix of ee pose.',
                        ),
                        'state_joint': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot 7-dof joint information (not used in orignial SAILOR dataset).',
                        ),
                        'state_gripper': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float32,
                            doc='Robot gripper opening width. Ranges between ~0 (closed) to ~0.077 (open)',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x ee relative pos, '
                            '3x ee relative rotation, 1x gripper action].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(f, demo_key, language_instruction, ds_name, ds_path, ac_type):
            # load raw data --> this should change for your dataset
            data = f["data/{}".format(demo_key)]
            episode_path = ds_path
            ep_len = len(data["actions"])

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(ep_len):
                # compute Kona language embedding
                language_embedding = self._embed([language_instruction])[0].numpy()

                state = np.concatenate((
                    data["obs"]["ee_pos"][i],
                    data["obs"]["ee_quat"][i],
                    data["obs"]["gripper_states"][i]
                ))

                if ac_type == "pos_gripper":
                    action = np.concatenate((
                        data['actions'][i][0:3], # pos
                        np.zeros(3), # 0 for rot
                        data['actions'][i][3:4], # gripper
                    ))
                elif ac_type == "pos_yaw_gripper":
                    action = np.concatenate((
                        data['actions'][i][0:3], # pos
                        np.zeros(2), # 0 for roll and pitch
                        data['actions'][i][3:4], # yaw
                        data['actions'][i][4:5], # gripper
                    ))
                elif ac_type == "pos_rot_gripper":
                    action = data['actions'][i]
                else:
                    raise ValueError

                episode.append({
                    'observation': {
                        'image': np.flip(data["obs"]['agentview_rgb'][i], axis=2),
                        'wrist_image': np.flip(data["obs"]['eye_in_hand_rgb'][i], axis=2),
                        'state': state.astype(np.float32),
                        'state_ee': data["obs"]["ee_states"][i].astype(np.float32),
                        'state_joint': data["obs"]["joint_states"][i].astype(np.float32),
                        'state_gripper': data["obs"]["gripper_states"][i].astype(np.float32),
                    },
                    'action': action.astype(np.float32),
                    'discount': 1.0,
                    'reward': float(i == (ep_len - 1)),
                    'is_first': i == 0,
                    'is_last': i == (ep_len - 1),
                    'is_terminal': i == (ep_len - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return "{}_{}".format(ds_name, demo_key), sample
        
        dataset_specs = dict(
            play=dict(
                path="/home/soroush/datasets/sailor_real/play/playdata128_0606_0607_0610.hdf5",
                language_instruction="Interact with the objects in diverse but meaningful ways.",
                ac_type="pos_rot_gripper",
            ),
            breakfast=dict(
                path="/home/soroush/datasets/sailor_real/breakfast/breakfast_hard_128_0608.hdf5",
                language_instruction="Place the bread, butter, and milk from the table onto the serving area.",
                ac_type="pos_rot_gripper",
            ),
            cook=dict(
                path="/home/soroush/datasets/sailor_real/cook/cook_medium_128_0613.hdf5",
                language_instruction="Place the fish, sausage, and tomato into the frying pan.",
                ac_type="pos_rot_gripper",
            ),
            cook_pan=dict(
                path="/home/soroush/datasets/sailor_real/cook/cook_hard_128_0611.hdf5",
                language_instruction="Place the pan onto the stove and place the fish and sausage into the pan.",
                ac_type="pos_rot_gripper",
            ),
        )

        for ds_name, ds_spec in dataset_specs.items():
            ds_path = ds_spec["path"]
            f = h5py.File(ds_path)
            language_instruction = ds_spec["language_instruction"]
            ac_type = ds_spec["ac_type"]
        
            # for smallish datasets, use single-thread parsing
            for demo_key in list(f["data"].keys()):
                yield _parse_example(
                    f, demo_key,
                    language_instruction=language_instruction,
                    ds_name=ds_name,
                    ds_path=ds_path,
                    ac_type=ac_type,
                )

            f.close()

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

