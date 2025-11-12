import os
import json
import pandas as pd
from datasets import Dataset

class RawDataset(object):
    def __init__(self, conf, logging):
        self.logging = logging
        self.path_dataset = os.path.join(conf.path.dataset, conf.dataprep.finetuning_dataset)
        self.rawdata_dir = os.path.join(conf.path.dataset, conf.dataprep.raw_dataset)
        self.subsample = conf.dataprep.subsample

    def prepare_SA_dataset(self):
        """task_SA_train.json과 task_SA_dev.json만 불러와 HuggingFace Dataset으로 변환"""
        for split in ['train', 'dev']:
            json_path = os.path.join(self.rawdata_dir, f"task_SA_{split}.json")
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"{json_path} 파일이 존재하지 않습니다.")

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            df = pd.DataFrame(data)
            dataset = Dataset.from_pandas(df)

            # text 필드 구성 (conversation + output)
            dataset = dataset.map(lambda x: {
                'text': f"""당신은 범죄 수사 전문가로, 주어진 대화를 읽고 보이스피싱 범행의 수법을 분석할 수 있습니다.
---
대화 : {x['conversation']}
분석 결과 : {x['output']}<|endoftext|>"""
            })

            save_path = os.path.join(self.path_dataset, f"task_SA_{split}_{self.subsample}.hf")
            dataset.save_to_disk(save_path)
            self.logging.info(f"[ScriptMind] Saved {split} dataset → {save_path}")
