import os
import pickle

import faiss
import pandas as pd

# local_rank = int(os.environ.get("LOCAL_RANK", 0))

# NQ320K
# split_length = {
#     -1: 0,
#     0: 98743,
#     1: 98743 + 2743,
#     2: 98743 + 2743 + 2743,
#     3: 98743 + 2743 + 2743 + 2743,
#     4: 98743 + 2743 + 2743 + 2743 + 2743,
# }

# MSMARCO
split_length = {
    -1: 0,
    0: 7957640,
    1: 7957640 + 221045,
    2: 7957640 + 221045 + 221045,
    3: 7957640 + 221045 + 221045 + 221045,
    4: 7957640 + 221045 + 221045 + 221045 + 221048,
}

def main():
    
    split = 1
    
    index = faiss.read_index("./mixlora_dsi/msmarco/d4/full_collection/full_dpr/mmap/model.index")

    text_ids = pd.read_csv("./mixlora_dsi/msmarco/d4/full_collection/full_dpr/mmap/text_ids.tsv", sep='\t',names=['id'])['id'].to_numpy()
    
    to_delete_text_ids = text_ids[split_length[split]:split_length[4]]
    text_ids = text_ids[:split_length[split]]
    
    index.remove_ids(to_delete_text_ids)
    
    print("length of index", index.ntotal)
    
    os.makedirs(f"./mixlora_dsi/msmarco/d{split}/full_collection/full_dpr", exist_ok=True)
    os.makedirs(f"./mixlora_dsi/msmarco/d{split}/full_collection/full_dpr/out", exist_ok=True)
    os.makedirs(f"./mixlora_dsi/msmarco/d{split}/full_collection/full_dpr/mmap", exist_ok=True)
    
    faiss.write_index(
        index, f"./mixlora_dsi/msmarco/d{split}/full_collection/full_dpr/mmap/model.index"
    )
    meta = {"text_ids": text_ids, "num_embeddings": len(text_ids)}
    print("meta data for index: {}".format(meta))

    with open(os.path.join(f"./mixlora_dsi/msmarco/d{split}/full_collection/full_dpr/mmap", "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    
    
    with open(os.path.join(
        f"./mixlora_dsi/msmarco/d{split}/full_collection/full_dpr/mmap", "text_ids.tsv"
    ), "w") as fout:
        for tid in text_ids:
            fout.write(f"{tid}\n")


if __name__ == "__main__":
    main()