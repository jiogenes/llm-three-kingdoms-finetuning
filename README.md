이 레포지토리는 블로그 포스트에서 사용된 전체 코드입니다.

---

Fine-tune Llama-3.2-1B-Instruct on filtered NamuWiki articles about 삼국지 using TRL’s SFTTrainer, Hugging Face Datasets, and PyTorch FSDP via 🤗 Accelerate.

---

## Quick start with uv

```command
uv init
uv venv --python 3.12 --seed
```

This creates a virtual environment (.venv) if none exists.

## Install dependencies

I have created a minimal requirements.txt (or copy/paste):

torch
torchvision
torchaudio
transformers
datasets
trl
accelerate
pyyaml

Then install with uv:

```command
uv pip install -r requirements.txt
```

Activate the virtual environment

```command
source ./.venv/bin/activate
```

## Config accelerate

Run the config wizard:

```command
accelerate config
```

The wizard writes a config file (often under ~/.cache/huggingface/accelerate/…).
Your script expects:

/your_home/.cache/huggingface/accelerate/default_config.yaml

CHANGE [THE CODE](https://github.com/jiogenes/llm-three-kingdoms-finetuning/blob/0ba833bbbb46b8ff843a922191ba166bd0a7ac28/finetuning.py#L51C16-L51C26) in the finetuning.py TO YOUR OWN PATH! 

## Run script

accelerate launch --num_processes 2 finetuning.py

## Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./three_kingdoms_sft/checkpoint-1000"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype='auto', device_map='auto')

messages = [
    {"role": "user", "content": "삼국지에 대해 알려줘."},
]

tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors='pt').to(model.device)
outputs = model.generate(**tokenized_chat, max_new_tokens=1024, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
>>>
Setting `pad_token_id` to `eos_token_id`:128009 for open-end generation.
system

Cutting Knowledge Date: December 2023
Today Date: 21 Oct 2025

user

삼국지에 대해 알려줘.assistant

중국의 2세기 말~3세기 말의 후한 말기와 삼국시대를 배경으로 한 진수의 역사서 정사 삼국지 혹은 나관중의 역사 소설 삼국지연의를 말한다. 이 둘은 이야기의 큰 줄기는 같지만 세세한 부분은 서로 다른 부분도 많다. 중국의 서기 184년 후한의 쇠퇴와 황건적의 난으로 인한 군웅할거 시대부터 사마염이 건국한 서진이 중국을 통일한 280년까지 있었던 역사를 다룬 책으로 이것의 소설판인 삼국지연의는 중국 4대 기서 중에서도 으뜸으로 치는 사람이 많으며 21세기인 현재에도 많은 사람들이 읽는 동아시아권을 대표하는 고전소설이다.
다만 삼국지를 말할 때 흔히 말하는 '정사'에는 정사 삼국지 이외에 삼국시대를 일부 다룬 후한서나 진서 등 몇몇 역사서의 내용을 포함해서 말하는 경우도 있으며, 후자의 경우는 나관중 이후 시대 삼국지연의 판본 오리지널 설정이나 화관색전 등이 포함되기도, 않기도 한다. 심지어는 20세기, 21세기에 창작된 설정이 후대 삼국지에 지속적으로 영향을 주기도 한다. 가령 도원결의 에피소드 시작 장면을 "황건적을 물리칠 의병을 구한다는 방문 앞에서 우연히 세 사람이 만났다"고 하면 기존 연의를 따른 것이고 "유비가 차(茶)를 사러 갔다 황건적 마원의를 만나 위기를 겪고 장비 덕에 목숨을 건진 뒤 장비에게 가보인 칼을 줬다가 어머니가 열 받아 차를 강물에 던져 버렸다"라고 하면 요시카와 에이지를 따른 본이다. 가장 최근의 사례로는 코에이사의 게임 삼국지 시리즈에서 시작된 수많은 설정들이 있다.
극소수는 영제의 즉위 때부터 치는 경우도 있다.
크게 진수가 쓴 정사 삼국지와 나관중의 소설 삼국지연의로 나뉜다. 정사는 말 그대로 '정확한 사실의 역사 기록'이며 연의는 소설이기 때문에 '실제 역사를 토대로 쓴 가상의 이야기'이다. 삼국지연의는 관우와 제갈량이 스타가 된 소설이기도 하며 연의 이후로 더욱 제갈량은 지혜의 화신 취급을 받고 관우는 관왕 혹은 관제라 불리며 무신(武神)이 되어 무속신앙의 대상이 되기도 한다. 여기에 다른 오호대장군도 정도는 다르지만 무속 신앙의 대상이 되었다. 근데 확실히 넘어가야 할 것은 관우가 관제묘로서 신으로 추앙받고 백성들이 유비 등을 추앙하고 제갈량 등을 지혜의 화신으로 섬긴 것은 연의보다 훨씬 이전이었다는 점이다. 추가로 많이 착각하지만 조조는 연의 이전에도 악역이었다. 서주 대학살이나, 병역을 2년에서 평생으로 늘린 둔전제 등의 영향으로 보인다. 즉, 연의 이전에도 이미 관우, 제갈량, 조조 등의 주요 인물들에 대한 평가가 이미 만들어져 있었고, 연의에서는 그러한 평가나 이미지를 극대화한 것으로 보아야 한다.
관우 신앙은 관우가 죽고 난 뒤 얼마 뒤부터 시작되어 이미 민간에선 상당한 입지를 가지고 있었다. 정확히 말하면 이미 관우의 입지가 대단하기에 삼국지연의에서 하늘로 올라가는 최후등이 추가되었다고 보는게 옳다. 물론 삼국지연의 후에 우주를 뚫을 기세가 된 것도 맞다. 한국 서울에 있는 동묘가 바로 관우를 모시는 사당이다.
이는 삼국시대가 끝나고부터 고작 몇십 년 이후의 사람인 석륵이 "장부가 일을 꾸미는 데 있어서는 마음이 호탕해서 일월과 같아야 하오. 짐은 조맹덕에 버금가는 길, '장부가 일을 꾸미는 것'이 바로 관우의 일생을 모르는 것이기 때문이다. 두보는 "장부가 일을 꾸미는
```
