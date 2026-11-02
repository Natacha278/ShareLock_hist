"""
GPU-resident sampler for the paired file written by build_quilt_pairs.py.

All features live on the GPU (a few GB), so a batch is just tensor indexing: no DataLoader.

    from quilt_pairs import QuiltPairs
    data = QuiltPairs("quilt_pairs_t05_biolord.pt", device="cuda")

    for epoch in range(n_epochs):
        for img, txt in data.epoch(batch_size=4096, split="train", by="image"):
            loss = model(img, txt) ...

    img_val, txt_val, gt = data.retrieval_set("val")   # for validation retrieval metrics

by="image": each epoch visits every image once and draws one of its captions at random.
            No image appears twice in a batch, so no false negatives from shared images,
            and images with many captions do not dominate.
by="pair" : each epoch visits every (image, caption) pair once (shuffled). An image can appear
            more than once in a batch.
"""
import torch


class QuiltPairs:
    def __init__(self, path, device="cuda", dtype=torch.float32):
        d = torch.load(path, map_location="cpu")
        self.meta = d["meta"]
        self.img_names, self.texts = d["img_names"], d["texts"]
        self.img = d["img_feats"].to(device=device, dtype=dtype)
        self.txt = d["txt_feats"].to(device=device, dtype=dtype)
        self.pair_img = d["pair_img"].to(device)
        self.pair_txt = d["pair_txt"].to(device)
        self.device = device

        # pairs are sorted by image: captions of image i are pair_txt[offsets[i]:offsets[i+1]]
        counts = torch.bincount(self.pair_img, minlength=self.img.shape[0])
        self.n_caps = counts
        self.offsets = torch.zeros(len(counts) + 1, dtype=torch.long, device=device)
        self.offsets[1:] = torch.cumsum(counts, 0)

        is_val = d["img_is_val"].to(device)
        self.images = {"train": (~is_val).nonzero().squeeze(1), "val": is_val.nonzero().squeeze(1),
                       "all": torch.arange(self.img.shape[0], device=device)}
        self.pairs = {k: torch.isin(self.pair_img, v).nonzero().squeeze(1) for k, v in self.images.items()}

    @property
    def dims(self):
        return self.img.shape[1], self.txt.shape[1]

    def random_caption(self, img_idx):
        """One random caption index (into self.txt) for each image in img_idx."""
        r = (torch.rand(len(img_idx), device=self.device) * self.n_caps[img_idx]).long()
        return self.pair_txt[self.offsets[img_idx] + r]

    def epoch(self, batch_size, split="train", by="image", drop_last=True):
        if by == "image":
            ids = self.images[split]
            ids = ids[torch.randperm(len(ids), device=self.device)]
        elif by == "pair":
            ids = self.pairs[split]
            ids = ids[torch.randperm(len(ids), device=self.device)]
        else:
            raise ValueError(by)
        stop = len(ids) - batch_size + 1 if drop_last else len(ids)
        for s in range(0, max(stop, 0), batch_size):
            b = ids[s:s + batch_size]
            if by == "image":
                yield self.img[b], self.txt[self.random_caption(b)]
            else:
                yield self.img[self.pair_img[b]], self.txt[self.pair_txt[b]]

    def retrieval_set(self, split="val"):
        """Image feats [I, D], caption feats [T, D] and a bool matrix gt[I, T] of correct matches."""
        imgs = self.images[split]
        pairs = self.pairs[split]
        txt_ids, txt_local = torch.unique(self.pair_txt[pairs], return_inverse=True)
        img_local = torch.searchsorted(imgs, self.pair_img[pairs])
        gt = torch.zeros(len(imgs), len(txt_ids), dtype=torch.bool, device=self.device)
        gt[img_local, txt_local] = True
        return self.img[imgs], self.txt[txt_ids], gt