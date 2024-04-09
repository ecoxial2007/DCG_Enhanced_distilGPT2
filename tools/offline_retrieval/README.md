# Step 1: Offline Report Retrieval

This step focuses on retrieving reports offline.

---

## 1. Generate Memo List
- `generate_memo_list.py`: Traverses the dataset and saves the information in a JSON list.

## 2. In-Memory Retrieval
- `in_memory.py`: Reads image features and stores them in a temporary folder. For each chest X-ray, it retrieves the top-k cases and saves them in `annotation_topk.json`.

**Note:** Currently, we provide only the key code snippets. You are welcome to modify our code to implement specific functionalities. A detailed tutorial will be updated later.
