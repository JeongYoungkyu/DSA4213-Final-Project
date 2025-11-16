def main():
  dataset = load_dataset("thu-coai/esconv")

  df_train = dataset['train']
  df_val = dataset['validation']

  def build_examples_from_esconv(conversations):
    rows = []
    for i in range(len(conversations)):
        r = conversations[i]
        data = json.loads(r["text"])
        
        dialog = data.get("dialog", [])
        situation = data.get("situation", "").strip()

        history = [
            "### System: You are a sensitive, non-clinical mental health assistant.",
        ]
        if situation:
            history.append(f"### Situation: {situation}")

        for turn in dialog:
            speaker = turn.get("speaker", "")
            txt = turn.get("text", "").strip()
            if not txt:
                continue

            if speaker == "usr":
                history.append(f"### User: {txt}")
            elif speaker == "sys":
                history.append(f"### Assistant: {txt}")

        # skip if we somehow only have the system line
        if len(history) <= 2:  # just System (and maybe Situation) = no dialog
            continue

        text = "\n".join(history) + "\n<|end|>"
        rows.append({"text": text})

    return Dataset.from_list(rows)

  ds_train = build_examples_from_esconv(df_train)
  ds_val = build_examples_from_esconv(df_val)

  def tokenize_function(batch):
    enc = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

  tokenized_ds_train = ds_train.map(tokenize_function, batched=True, remove_columns=["text"])
  tokenized_ds_val = ds_val.map(tokenize_function, batched=True, remove_columns=["text"])
if __name__ == "__main__":
    main()
