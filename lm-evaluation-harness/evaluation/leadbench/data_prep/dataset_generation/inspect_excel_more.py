import pandas as pd

try:
    df = pd.read_excel('/data1/yezj/gitlab/leadbench/tmp/data/normal_anti_hijack_abc_stage2_6_49段.xlsx')
    print("Unique Roles:", df['role'].unique())
    print("Unique Rounds:", df['round'].unique())
    
    # Get one full dialog
    dialog_ids = df['dialog_id'].unique()
    if len(dialog_ids) > 0:
        first_dialog_id = dialog_ids[0]
        dialog_df = df[df['dialog_id'] == first_dialog_id].sort_values('sentence_id')
        print(f"\nDialog {first_dialog_id}:")
        for _, row in dialog_df.iterrows():
            print(f"[{row['role']}] (Round {row['round']}): {row['sentence']}")

except Exception as e:
    print(f"Error: {e}")
