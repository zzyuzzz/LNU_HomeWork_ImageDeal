from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, face:Dataset, noface:Dataset) -> None:
        super().__init__()
        self.face = face
        self.noface = noface

    def __getitem__(self, index):
        face_len = len(self.face)
        if index >= face_len:
            return self.noface[index-face_len]
        return self.face[index]
    
    def __len__(self):
        return len(self.face)+len(self.noface)