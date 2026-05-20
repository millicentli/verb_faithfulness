import nnsight
import torch
import vec2text
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", attn_implementation="flash_attention_2", device_map="auto", torch_dtype=torch.bfloat16)
model = nnsight.LanguageModel(model, tokenizer=tokenizer)
model.cuda()

device = "cuda"
inversion_model=vec2text.models.InversionFromLogitsEmbModel.from_pretrained("jxm/t5-base__llama-7b__one-million-instructions__emb").to(device)
corrector_model = vec2text.models.CorrectorEncoderFromLogitsModel.from_pretrained("jxm/t5-base___llama-7b___one-million-instructions__correct")
corrector = vec2text.load_corrector(inversion_model,corrector_model)
# inversion_model = vec2text.models.InversionModel.from_pretrained("jxm/gtr__nq__32")
# corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained("jxm/gtr__nq__32__correct")
# corrector = vec2text.load_corrector(inversion_model, corrector_model)

# prompt = "who is the president of the united states?"
# LAYER_IDX = 15
# with model.trace(prompt) as tracer:
#     if hasattr(model, 'model'):
#         clean_hs = model.model.layers[LAYER_IDX].output[0][:, -1, :].save()
#     else:
#         clean_hs = model.transformer.h[LAYER_IDX].output[0][:, -1, :].save()


import openai

def get_embeddings_openai(text_list, model="text-embedding-ada-002") -> torch.Tensor:
    outputs = []
    client = openai.OpenAI()
    response = client.embeddings.create(
        input=text_list,
        model=model,
        encoding_format="float",  # override default base64 encoding...
    )
    # outputs.extend([e["embedding"] for e in response["data"]])
    # outputs.extend([response.data[0].embedding])
    outputs.extend([e.embedding for e in response.data])
    return torch.tensor(outputs)


embeddings = get_embeddings_openai([
       "Jack Morris is a PhD student at Cornell Tech in New York City",
       "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity"
])
breakpoint()


print(vec2text.invert_embeddings(
    embeddings=embeddings.cuda(),
    corrector=corrector
))

# output = vec2text.invert_embeddings(
#     embeddings=clean_hs.cuda(),
#     corrector=corrector
# )

# print(output)