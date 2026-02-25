import torch


class Qwen3VLDataCollator():
    def __init__(self, processor):
        self.processor = processor

    def _clean_message(self, message):
        """
        remove unnecessary keys from message(very very necessary)
        """
        out_message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video", 
                        "video": message[0]["content"][0]["video"], 
                        "max_pixels": message[0]["content"][0]["max_pixels"], 
                        "fps": message[0]["content"][0]["fps"] if "fps" in message[0]["content"][0] else None,
                        "nframes": message[0]["content"][0]["nframes"] if "nframes" in message[0]["content"][0] else None,
                        "sample_type": message[0]["content"][0]["sample_type"] if "sample_type" in message[0]["content"][0] else "uniform",
                    },
                    {
                        "type": "text", 
                        "text": message[0]["content"][1]["text"]
                    },
                ],
            }
        ]

        if out_message[0]["content"][0]["fps"] is None:
            out_message[0]["content"][0].pop("fps")
        if out_message[0]["content"][0]["nframes"] is None:
            out_message[0]["content"][0].pop("nframes")
        
        return out_message

    def __call__(self, features):
        """
        Preprocess inputs to token sequences and return a batch
        """
        features_A = []
        features_B = []
        has_idx = "metainfo_idx" in features[0] and features[0]["metainfo_idx"] is not None

        for feature in features:
            features_A.append(self._clean_message(feature["A_data"]))
            features_B.append(self._clean_message(feature["B_data"]))
        
        batch_A = self.processor.apply_chat_template(
            features_A,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
        )
        batch_B = self.processor.apply_chat_template(
            features_B,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
        )

        chosen_label = torch.stack([torch.tensor(feature["chosen_label"]) for feature in features])
        A_scores = torch.stack([torch.tensor(feature["A_scores"]) for feature in features])
        B_scores = torch.stack([torch.tensor(feature["B_scores"]) for feature in features])
        
        batch = {
            "A": batch_A,
            "B": batch_B,
            "return_loss": True,
            "chosen_label": chosen_label,
            "A_scores": A_scores,
            "B_scores": B_scores,
        }

        if has_idx:
            metainfo_idx = torch.stack([torch.tensor(feature["metainfo_idx"]) for feature in features])
            batch["metainfo_idx"] = metainfo_idx

        return batch