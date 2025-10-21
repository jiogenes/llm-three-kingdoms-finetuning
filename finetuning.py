import re
import yaml

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def remove_trailing(text: str) -> str:
    """
    문자열 끝에 '(삼국지)'가 붙어 있으면 제거합니다.
    예: '조조전 (삼국지)' -> '조조전'
    """
    if text is None:
        return text
    return re.sub(r'\s*\(삼국지\)\s*$', '', text)

def preprocess_dataset():
    # Load dataset
    dataset = load_dataset("heegyu/namuwiki-extracted", split="train")

    include_title_keywords = ['삼국지', '조조', '유비', '손권', '제갈량', '관우', '장비', '조운', '여포']
    include_text_keywords =  ["삼국지", "삼국지연의", "삼국지(연의)", "삼국지(정사)", "삼국지정사", "나관중", "삼국지 연의", "삼국지 정사"]
    except_title_keywords = ['삼국지W', '삼국지조조전', '삼국지 조조전', '삼국지 시리즈', '기타 창작물', 'DS', '디펜스', '삼국지대전', '영걸전', '영웅 삼국지', '영웅삼국지', '시, 연 삼국지화', '아이돌전', '무징쟁패', '명장 관우', '용의 부활', '백화난무', '소녀대전', '카츠마타', '쾌도난담', '헬로우', 'PK', '모던픽션', '마행처우역거', '화평자전', '곱빼기', '불편한', '건담', '시뮬레이터', '유비트', '가족 삼국지', '요시카와', '개그', '세븐나이츠', '레이예스', '퍼즐앤드래곤', '선녀강림', '황제의 반란', '조조삼소', '레전드히어로', '삼국전투기', '진삼국무쌍', '천도의 주인', '천명', '연희 시리즈', '매직 더 개더링', '같은 꿈을 꾸다', '강철삼국지', '무장쟁패', '삼국지 도원결의', '삼국지 갤러리', '레전드히어로', '와이파이', '조조전 Cost 계산기', '삼국지톡', '여포키우기', '왕자영요', '토탈 워', '삼국지 책략전', '삼국지 조운장군전', '노래하는유비', '며느리 삼국지', '하이스쿨', '요괴', '고우영 삼국지', '양지운의 라디오 삼국지', '한권으로 읽는 소설 삼국지', '영웅조조', '배틀필드', '여포양', '장비 압수', '요코야마', '톨기스', '신조조전', '소노다', '삼국지퍼즐대전', '삼국지평화', '여자 제갈량']
    except_title_keywords += ['삼국지 ' + str(i) for i in range(1, 16)]
    keep_indices = []
    for i, item in enumerate(dataset):
        if any(keyword in item['title'] for keyword in include_title_keywords) and any(keyword in item['text'] for keyword in include_text_keywords):
            if any(keyword in item['title'] for keyword in except_title_keywords):
                continue
            keep_indices.append(i)

    filtered_dataset = dataset.select(keep_indices)
    filtered_dataset = filtered_dataset.map(lambda x: {'messages': [
        {'role': 'user', 'content': f"삼국지에서 {remove_trailing(x['title'])}에 대해 알려줘."},
        {'role': 'assistant', 'content': f"{x['text']}"}
    ]}, remove_columns=['text', 'title', 'contributors', 'namespace'])

    return filtered_dataset


def main():
    # Preprocess dataset
    dataset = preprocess_dataset()

    # Configure model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    with open('/home/jyji/.cache/huggingface/accelerate/default_config.yaml', "r") as f:
        fsdp_config = yaml.safe_load(f)

    # Configure trainer
    training_args = SFTConfig(
        output_dir="./three_kingdoms_sft",
        overwrite_output_dir=True,
        max_steps=2000,
        per_device_train_batch_size=1,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        packing=True,
        fsdp="full_shard",
        fsdp_config=fsdp_config,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Start training
    trainer.train(resume_from_checkpoint=False)

    # 로그 기록 저장
    logs = trainer.state.log_history
    with open("training_logs.pt", "wb") as f:
        torch.save(logs, f)

if __name__ == "__main__":
    main()