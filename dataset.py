import torch
class FaceVideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split="train", val_ratio=0.05, num_frames=16):
        data = torch.load(data_path)
        self.latents_list = data['latents']
        self.meta = data['meta']
        self.split = split
        self.val_ratio = val_ratio
        self.num_frames = num_frames

        n = len(self.latents_list)
        n_val = int(val_ratio*n)
        if split == 'train':
            self.latents_list = self.latents_list[n_val:]
            self.meta = self.meta[n_val:]
        else:
            self.latents_list = self.latents_list[:n_val]
            self.meta = self.meta[:n_val]

    def __getitem__(self, idx):
        # returns (latents, emotion_idx)
        # latents: (16, 4, 32, 32) float32
        # emotion_idx: int
        all_latents = self.latents_list[idx]
        num_frames = all_latents.shape[0]
        start = torch.randint(0, num_frames - self.num_frames + 1, (1,)).item()
        latents = all_latents[start:start+self.num_frames, :, :, :].float()
        emotion_idx = self.meta[idx]["emotion_idx"]
        return latents, emotion_idx

    def __len__(self):
        return len(self.latents_list)