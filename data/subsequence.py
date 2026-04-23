import random
import torch

class RepeatedSubsequenceDataset:
    def __init__(
        self,
        vocab_size=50,
        seq_len=64,
        subseq_len=8,
        num_samples=512,   # set to None for "infinite data"
        seed=42,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.subseq_len = subseq_len
        self.num_samples = num_samples
        random.seed(seed)

        if num_samples is not None:
            self.data = [self._generate_example() for _ in range(num_samples)]
        else:
            self.data = None  # infinite mode

    def _generate_example(self):
        # 1. Create a random sequence
        seq = [random.randint(1, self.vocab_size - 1) for _ in range(self.seq_len)]

        # 2. Choose a subsequence
        start = random.randint(0, self.seq_len - 2 * self.subseq_len - 1)
        subseq = seq[start : start + self.subseq_len]

        # 3. Insert repeated subsequence later
        repeat_start = random.randint(start + self.subseq_len, self.seq_len - self.subseq_len)
        seq[repeat_start : repeat_start + self.subseq_len] = subseq

        # Convert to tensor
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)

        return x, y

    def __len__(self):
        return self.num_samples if self.num_samples is not None else 10**12

    def __getitem__(self, idx):
        if self.data is not None:
            return self.data[idx]
        else:
            return self._generate_example()


# --- Example usage ---

if __name__ == "__main__":
    # Finite dataset (like "512 data points")
    dataset = RepeatedSubsequenceDataset(num_samples=512)

    x, y = dataset[0]
    print("Input:", x)
    print("Target:", y)

    # Infinite data version
    infinite_dataset = RepeatedSubsequenceDataset(num_samples=None)
    x_inf, y_inf = infinite_dataset[0]
    print("Infinite sample:", x_inf)