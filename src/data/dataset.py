import torch
from datasets import load_dataset, DatasetDict
from src.data.template import build_prompt


def convert_GSB_csv_to_reward_data(
    example, 
    data_dir, 
    eval_dims=["VQ"], 
    max_pixels=448 * 448, 
    fps=2.0, 
    num_frames=None, 
    prompt_template_type="none", 
    sample_type="uniform"
):
    """
    Convert Good/Same/Bad csv data to reward data.

    Args:
        example (dict): A dataframe containing the GSB csv data.
        data_dir (str): The directory path to the video files.
        eval_dim (str): The dimension to evaluate ("VQ"/"MQ"/"TA").
        max_pixels (int): The maximum number of pixels allowed for videos.
        num_frames (float): Number of frames.
        prompt_template_type (str): The type of prompt template to use ("none"/"simple"/"video_score").

    Returns:
        dict: A dictionary containing the reward data.
    """

    A_data = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video", 
                    "video": f"{data_dir}/{example[f'path_A']}", 
                    "max_pixels": max_pixels, 
                    "fps": fps if num_frames is None else None,
                    "nframes": min(num_frames, example[f"num_frames_A"]) if num_frames is not None else None,
                    "sample_type": sample_type,
                },
                {"type": "text", "text": build_prompt(example["prompt"], eval_dims, prompt_template_type)},
            ],
        }
    ]
    B_data = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video", 
                    "video": f"{data_dir}/{example[f'path_B']}", 
                    "max_pixels": max_pixels, 
                    "fps": fps if num_frames is None else None,
                    "nframes": min(num_frames, example[f"num_frames_B"]) if num_frames is not None else None,
                    "sample_type": sample_type,
                },
                {"type": "text", "text": build_prompt(example["prompt"], eval_dims, prompt_template_type)},
            ],
        }
    ]

    chosen_labels = []
    A_scores = []
    B_scores = []
    
    for eval_dim in eval_dims:
        ### chosen_label: 1 if A is chosen, -1 if B is chosen, 0 if tied.
        ### 22 if invalid. ooaaeeaa o.O
        try:
            if example[f"{eval_dim}"] is not None:
                if example[f"{eval_dim}"] == "A":
                    chosen_label = 1
                elif example[f"{eval_dim}"] == "B":
                    chosen_label = -1
                elif example[f"{eval_dim}"] == "same":
                    chosen_label = 0
                elif example[f"{eval_dim}"] == "invalid":
                    chosen_label = 22
                else:
                    chosen_label = 22
            else:
                chosen_label = 22
        except Exception as e:
            chosen_label = 22

        chosen_labels.append(chosen_label)
        if f"MOS_A_{eval_dim}" in example and f"MOS_B_{eval_dim}" in example:
            try:
                A_score = example[f"MOS_A_{eval_dim}"] if example[f"MOS_A_{eval_dim}"] is not None else 0.0
                B_score = example[f"MOS_B_{eval_dim}"] if example[f"MOS_B_{eval_dim}"] is not None else 0.0
            except Exception as e:
                A_score = 0.0
                B_score = 0.0
            A_scores.append(A_score)
            B_scores.append(B_score)
        else:
            A_scores.append(0.0)
            B_scores.append(0.0)

    chosen_labels = torch.tensor(chosen_labels, dtype=torch.long)
    A_scores = torch.tensor(A_scores, dtype=torch.float)
    B_scores = torch.tensor(B_scores, dtype=torch.float)
    metainfo_idx = None
    if 'metainfo_idx' in example:
        metainfo_idx = example['metainfo_idx']

    ret = dict(
        A_data=A_data,
        B_data=B_data,
        A_scores=A_scores,
        B_scores=B_scores,
        chosen_label=chosen_labels,
        metainfo_idx=metainfo_idx,
    )
    return ret