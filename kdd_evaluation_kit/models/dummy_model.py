import os
import torch
import json
import random
import faiss
import re
import vllm

import torch.nn.functional as F
from typing import List, Union, Any, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np
from FlagEmbedding import FlagReranker

from .base_model import ShopBenchBaseModel
# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 42*4096))



AICROWD_SUBMISSION_BATCH_SIZE = 8 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 2 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.


class llama3_8b_FewShot_vllm(ShopBenchBaseModel):
    """
    A dummy model implementation for ShopBench, illustrating how to handle both
    multiple choice and other types of tasks like Ranking, Retrieval, and Named Entity Recognition.
    This model uses a consistent random seed for reproducible results.
    """

    def __init__(self):
        """Initializes the model and sets the random seed for consistency."""
        random.seed(AICROWD_RUN_SEED)
        self.initialize_models()

    def initialize_models(self):
        # Initialize Meta Llama 3 - 8B Instruct Model
        self.model_name = "./models/Meta-Llama-3-8B-Instruct"

        if not os.path.exists(self.model_name):
            raise Exception(
                f"""
            The evaluators expect the model weights to be checked into the repository,
            but we could not find the model weights at {self.model_name}
            
            Please follow the instructions in the docs below to download and check in the model weights.
                https://gitlab.aicrowd.com/aicrowd/challenges/amazon-kdd-cup-2024/amazon-kdd-cup-2024-starter-kit/-/blob/master/docs/download-baseline-model-weights.md
            
            """
            )

        # initialize the model with vllm
        self.system_prompt = "You are a helpful and multilingual online shopping assistant. You can understand and respond to user queries in English, German, Italian, French, Japanese, Spanish, Chinese. You are knowledgeable about various products. NOTE:ONLY OUTPUT THE ANSWER!!\n\n"

        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            trust_remote_code=True,
            dtype="half", # note: bfloat16 is not supported on nvidia-T4 GPUs
            enforce_eager=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            # self.tokenizer.convert_tokens_to_ids("\\n"),
        ]

        self.faiss = self.load_rag_module(faiss_index_path="./models/large_index.ivf")
        self.faiss_retrieve_topk = 8
        self.faiss_score_filter = 0.85
        self.bge_rerank_topk = 6
        self.bge_score_filter = 0.6

        self.retrieve_task_description = "Given a online shopping user query, retrieve relevant Question-Answer that similar (type of task ,languages involved and product) to the query."

    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
        
    def load_rag_module(self, faiss_index_path: str):
        # rag_module : embedding + faiss index + reranker
        self.embed_model = SentenceTransformer("./models/multilingual-e5-large-instruct", device='cpu')
        # self.reranker = FlagReranker('./models/bge-reranker-v2-m3', use_fp16=True, device='cuda:1')

        # few shot preprocess
        self.few_shot_example_text = []
        with open('./models/large_sample_example.jsonl','r',encoding='utf8') as f:
            for i in f.readlines():
                passage = ''
                t_data = json.loads(i.strip())
                if "input" in t_data:
                    passage = t_data['instruction'] + t_data['input'] + '\nOutput:' + str( t_data['output']) + '\n'
                else:
                    passage = t_data['instruction'] + str(t_data['output']) + '\n'
                passage = passage.replace('\\n','\n')
                self.few_shot_example_text.append(passage)
        if os.path.exists(faiss_index_path):
            self.index = faiss.read_index(faiss_index_path)
        else:
            self.index = self.train_save_faiss_index(faiss_index_path)
        self.metadata = [{"fewshot_examaple": fewshot_examaple} for fewshot_examaple in self.few_shot_example_text]


    def train_save_faiss_index(self,
                               index_save_path: str = "./models/index.ivf",
                               dim: int = 1024,
                               nlist: int = 1024,
                               index_nprobe: int = 3):
        # preprocess train retrieve index and save trained index
        # dim : Embedding dimension for intfloat/multilingual-e5-large
        # nlist : Number of cluster centroids
        fewshot_embeddings = []
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = index_nprobe
        fewshot_embeddings = self.embed_model.encode(self.few_shot_example_text, batch_size=256+128, show_progress_bar=True)
        print(f'process few shot example embedding done! {len(self.few_shot_example_text)}')
        index.train(fewshot_embeddings.astype(np.float32))
        index.add(fewshot_embeddings.astype(np.float32))
        faiss.write_index(index, index_save_path)
        del fewshot_embeddings

        return index


    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_predict` function.

        Returns:
            int: The batch size, an integer between 1 and 16. This value indicates how many
                 queries should be processed together in a single batch. It can be dynamic
                 across different batch_predict calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE
        return self.batch_size

    def batch_predict(self, batch: Dict[str, Any], is_multiple_choice:bool) -> List[str]:
        """
        Generates a batch of prediction based on associated prompts and task_type

        For multiple choice tasks, it randomly selects a choice.
        For other tasks, it returns a list of integers as a string,
        representing the model's prediction in a format compatible with task-specific parsers.

        Parameters:
            - batch (Dict[str, Any]): A dictionary containing a batch of input prompts with the following keys
                - prompt (List[str]): a list of input prompts for the model.
    
            - is_multiple_choice bool: A boolean flag indicating if all the items in this batch belong to multiple choice tasks.

        Returns:
            str: A list of predictions for each of the prompts received in the batch.
                    Each prediction is
                           a string representing a single integer[0, 3] for multiple choice tasks,
                        or a string representing a comma separated list of integers for Ranking, Retrieval tasks,
                        or a string representing a comma separated list of named entities for Named Entity Recognition tasks.
                        or a string representing the (unconstrained) generated response for the generation tasks
                        Please refer to parsers.py for more details on how these responses will be parsed by the evaluator.
        """
        prompts = batch["prompt"]
        
        # format prompts using the chat template
        formatted_prompts = self.format_prommpts(prompts, is_multiple_choice)
        # set max new tokens to be generated
        max_new_tokens = 128 
        
        if is_multiple_choice:
            max_new_tokens = 1 # For MCQ tasks, we only need to generate 1 token
        
        
        # Generate responses via vllm
        responses = self.llm.generate(
            formatted_prompts,
            vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                # top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                # top_k=1,
                temperature=0,  # randomness of the sampling
                seed=AICROWD_RUN_SEED, # Seed for reprodicibility
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=max_new_tokens,  # Maximum number of tokens to generate per output sequence.
                # stop_token_ids=self.terminators # llama 3 stop token
            ),
            use_tqdm = False
        )
        # Aggregate answers into List[str]
        batch_response = []
        for response in responses:
            batch_response.append(response.outputs[0].text)        
            
        if is_multiple_choice:
            print("MCQ: ", batch_response)

        return batch_response

    def format_prommpts(self, prompts:List[str], is_multiple_choice:bool):
        """
        Formats prompts using the chat_template of the model.
            
        Parameters:
        - queries (list of str): A list of queries to be formatted into prompts.
            
        """
        
        # 1. faiss index retrieve topK few shot example
        # 2. rerank few shot example
        # 3. select topK few shot example as prompt
        # 4.  [
        #       {"role":"system","content":self.system_prompt},
        #       {"role":"user","content": query + few shot exmaple}
        #       {"role":"assistant","content": model generate ... ...}
        #  ]

        # faiss vector Retrieve smiliar few shot example
        formatted_prompts = []
        for prompt in prompts:
            query_text = ' ' + prompt
            query_embed = self.embed_model.encode([self.get_detailed_instruct(self.retrieve_task_description, query_text)])[0]
            scores, indices = self.index.search(np.array([query_embed]).astype(np.float32), self.faiss_retrieve_topk)
            # process results
            few_shot_exmaple = []
            for score, retrieved_idx in zip(scores[0], indices[0]):
                if score>=self.faiss_score_filter and len(self.metadata[retrieved_idx]["fewshot_examaple"])<=6000:
                    fewshot_examaple = self.metadata[retrieved_idx]["fewshot_examaple"]
                    few_shot_exmaple.append(fewshot_examaple)
            
            
            reranked_exmaple_prompt = few_shot_exmaple[:4] if  is_multiple_choice else few_shot_exmaple



            # rerank


            # if len(exmaple_prompt)>0:
            #     print("before reranke:")
            #     print(exmaple_prompt[:4])
            #     # rerank the result
            #     retrank_task_description = "Given a online shopping user query, retrieve relevant Question-Answer that same (type of task ,languages involved) to the query."
            #     rerank_scores = self.reranker.compute_score(
            #         [[query_text,retrieved_fse] for retrieved_fse in exmaple_prompt],
            #         normalize=True,
            #         batch_size=32
            #     )
            #     reranked_exmaple_prompt = [
            #         exmaple_prompt[bge_rerank_topK_idx] 
            #             for bge_rerank_topK_idx in np.argsort(rerank_scores)[-bge_rerank_topk:] 
            #             if rerank_scores[bge_rerank_topK_idx]>=bge_score_filter
            #     ]
            #     reranked_exmaple_prompt = [t_prompt for t_prompt in reranked_exmaple_prompt if len(t_prompt)<=6000]
            #     print("reranked:")
            # else:
            #     reranked_exmaple_prompt = []
            
            if len(reranked_exmaple_prompt) > 0:
                prompt_example = '## Here are some similar questions and answers you can refer to:\n' 
                for i in reranked_exmaple_prompt:
                    prompt_example += i+'\n'
                prompt_example += '## Now answer the Question:' + prompt
            else:
                prompt_example = '## Now answer the Question:' + prompt

                
            if is_multiple_choice:
                chat_prompt = self.system_prompt + prompt_example
            else:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt_example}
                ]
                chat_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    # return_tensors="pt"
                )
            chat_prompt = chat_prompt[len("<|begin_of_text|>"):]
            formatted_prompts.append(chat_prompt)

        return formatted_prompts




class aya23_fewshot_VLLM(llama3_8b_FewShot_vllm):
    """
    A dummy model implementation for ShopBench, illustrating how to handle both
    multiple choice and other types of tasks like Ranking, Retrieval, and Named Entity Recognition.
    This model uses a consistent random seed for reproducible results.
    """

    def __init__(self):
        """Initializes the model and sets the random seed for consistency."""
        random.seed(AICROWD_RUN_SEED)
        self.initialize_models()

    def initialize_models(self):
        # Initialize Meta Llama 3 - 8B Instruct Model
        self.model_name = "./models/aya23-8b"

        if not os.path.exists(self.model_name):
            raise Exception(
                f"""
            The evaluators expect the model weights to be checked into the repository,
            but we could not find the model weights at {self.model_name}
            
            Please follow the instructions in the docs below to download and check in the model weights.
                https://gitlab.aicrowd.com/aicrowd/challenges/amazon-kdd-cup-2024/amazon-kdd-cup-2024-starter-kit/-/blob/master/docs/download-baseline-model-weights.md
            
            """
            )

        # initialize the model with vllm
        self.system_prompt = "You are a helpful and multilingual online shopping assistant. You can understand and respond to user queries in English, German, Italian, French, Japanese, Spanish, Chinese. You are knowledgeable about various products. NOTE:ONLY OUTPUT THE ANSWER!!\n\n"

        self.model = AutoModelForCausalLM.from_pretrained("/home/jnu/gxw/amazon-kdd-cup-2024-starter-kit/models/aya23-8b", torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained("/home/jnu/gxw/amazon-kdd-cup-2024-starter-kit/models/aya23-8b")

        self.faiss = self.load_rag_module(faiss_index_path="./models/large_index.ivf")
        self.faiss_retrieve_topk = 8
        self.faiss_score_filter = 0.85
        self.bge_rerank_topk = 6
        self.bge_score_filter = 0.6

        self.retrieve_task_description = "Given a online shopping user query, retrieve relevant Question-Answer that similar (type of task ,languages involved and product) to the query."

    @torch.no_grad()
    def batch_predict(self, batch: Dict[str, Any], is_multiple_choice: bool) -> List[str]:

        prompts = batch["prompt"]
        
        # format prompts using the chat template
        formatted_prompts = self.format_prommpts(prompts, is_multiple_choice)
        # set max new tokens to be generated
        max_new_tokens = 140
        if is_multiple_choice:
            max_new_tokens = 1
        input_batch = [self.tokenizer.encode(i,return_tensors="pt").to(1) for i in formatted_prompts]

        
        # Generate responses via vllm
        gen_tokens = [self.model.generate(
            i,
            do_sample=False,
            max_new_tokens=max_new_tokens,  # Maximum number of tokens to generate per output sequence.
        ) for i in input_batch]

        gen_tokens = [i[0][len(j[0]):] for i,j in zip(gen_tokens, input_batch)]
        gen_text = [self.tokenizer.decode(i,skip_special_tokens=True) for i in gen_tokens]

        if is_multiple_choice:
            print("MCQ: ", gen_text)
        else:
            print(gen_text)
        del input_batch, gen_tokens
        torch.cuda.empty_cache()
        return gen_text

    def format_prommpts(self, prompts:List[str], is_multiple_choice:bool):
        """
        Formats prompts using the chat_template of the model.
            
        Parameters:
        - queries (list of str): A list of queries to be formatted into prompts.
            
        """
        
        # 1. faiss index retrieve topK few shot example
        # 2. rerank few shot example
        # 3. select topK few shot example as prompt
        # 4.  [
        #       {"role":"system","content":self.system_prompt},
        #       {"role":"user","content": query + few shot exmaple}
        #       {"role":"assistant","content": model generate ... ...}
        #  ]

        # faiss vector Retrieve smiliar few shot example
        formatted_prompts = []
        for prompt in prompts:
            query_text = ' ' + prompt
            query_embed = self.embed_model.encode([self.get_detailed_instruct(self.retrieve_task_description, query_text)])[0]
            scores, indices = self.index.search(np.array([query_embed]).astype(np.float32), self.faiss_retrieve_topk)
            # process results
            few_shot_exmaple = []
            for score, retrieved_idx in zip(scores[0], indices[0]):
                if score>=self.faiss_score_filter and len(self.metadata[retrieved_idx]["fewshot_examaple"])<=6000:
                    fewshot_examaple = self.metadata[retrieved_idx]["fewshot_examaple"]
                    few_shot_exmaple.append(fewshot_examaple)
            
            
            reranked_exmaple_prompt = few_shot_exmaple if  is_multiple_choice else few_shot_exmaple[:4]



            # rerank


            # if len(exmaple_prompt)>0:
            #     print("before reranke:")
            #     print(exmaple_prompt[:4])
            #     # rerank the result
            #     retrank_task_description = "Given a online shopping user query, retrieve relevant Question-Answer that same (type of task ,languages involved) to the query."
            #     rerank_scores = self.reranker.compute_score(
            #         [[query_text,retrieved_fse] for retrieved_fse in exmaple_prompt],
            #         normalize=True,
            #         batch_size=32
            #     )
            #     reranked_exmaple_prompt = [
            #         exmaple_prompt[bge_rerank_topK_idx] 
            #             for bge_rerank_topK_idx in np.argsort(rerank_scores)[-bge_rerank_topk:] 
            #             if rerank_scores[bge_rerank_topK_idx]>=bge_score_filter
            #     ]
            #     reranked_exmaple_prompt = [t_prompt for t_prompt in reranked_exmaple_prompt if len(t_prompt)<=6000]
            #     print("reranked:")
            # else:
            #     reranked_exmaple_prompt = []
            
            if len(reranked_exmaple_prompt) > 0:
                prompt_example = '## Here are some similar questions and answers you can refer to:\n' 
                for i in reranked_exmaple_prompt:
                    prompt_example += i+'\n'
                prompt_example += '## Now answer the Question:' + prompt
            else:
                prompt_example = '## Now answer the Question:' + prompt


            messages = [
                {"role": "user", "content": prompt_example}
            ]
            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                # return_tensors="pt"
            )

            formatted_prompts.append(chat_prompt)

        return formatted_prompts










class DummyModel(ShopBenchBaseModel):
    """
    A dummy model implementation for ShopBench, illustrating how to handle both
    multiple choice and other types of tasks like Ranking, Retrieval, and Named Entity Recognition.
    This model uses a consistent random seed for reproducible results.
    """

    def __init__(self):
        """Initializes the model and sets the random seed for consistency."""
        random.seed(AICROWD_RUN_SEED)

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_predict` function.

        Returns:
            int: The batch size, an integer between 1 and 16. This value indicates how many
                 queries should be processed together in a single batch. It can be dynamic
                 across different batch_predict calls, or stay a static value.
        """
        self.batch_size = 4
        return self.batch_size

    def batch_predict(self, batch: Dict[str, Any], is_multiple_choice:bool) -> List[str]:
        """
        Generates a batch of prediction based on associated prompts and task_type

        For multiple choice tasks, it randomly selects a choice.
        For other tasks, it returns a list of integers as a string,
        representing the model's prediction in a format compatible with task-specific parsers.

        Parameters:
            - batch (Dict[str, Any]): A dictionary containing a batch of input prompts with the following keys
                - prompt (List[str]): a list of input prompts for the model.
    
            - is_multiple_choice bool: A boolean flag indicating if all the items in this batch belong to multiple choice tasks.

        Returns:
            str: A list of predictions for each of the prompts received in the batch.
                    Each prediction is
                           a string representing a single integer[0, 3] for multiple choice tasks,
                        or a string representing a comma separated list of integers for Ranking, Retrieval tasks,
                        or a string representing a comma separated list of named entities for Named Entity Recognition tasks.
                        or a string representing the (unconstrained) generated response for the generation tasks
                        Please refer to parsers.py for more details on how these responses will be parsed by the evaluator.
        """
        prompts = batch["prompt"]

        possible_responses = [1, 2, 3, 4]

        batch_response = []
        for prompt in prompts:
            if is_multiple_choice:
                # Randomly select one of the possible responses for multiple choice tasks
                batch_response.append(str(random.choice(possible_responses)))
            else:
                # For other tasks, shuffle the possible responses and return as a string
                random.shuffle(possible_responses)
                batch_response.append(str(possible_responses))
                # Note: As this is dummy model, we are returning random responses for non-multiple choice tasks.
                # For generation tasks, this should ideally return an unconstrained string.

        return batch_response


# class DummyModel(ShopBenchBaseModel):
#     """
#     A dummy model implementation for ShopBench, illustrating how to handle both
#     multiple choice and other types of tasks like Ranking, Retrieval, and Named Entity Recognition.
#     This model uses a consistent random seed for reproducible results.
#     """

#     def __init__(self):
#         """Initializes the model and sets the random seed for consistency."""
#         random.seed(AICROWD_RUN_SEED)

#     def predict(self, prompt: str, is_multiple_choice: bool) -> str:
#         """
#         Generates a prediction based on the input prompt and task type.

#         For multiple choice tasks, it randomly selects a choice.
#         For other tasks, it returns a list of integers as a string,
#         representing the model's prediction in a format compatible with task-specific parsers.

#         Args:
#             prompt (str): The input prompt for the model.
#             is_multiple_choice (bool): Indicates whether the task is a multiple choice question.

#         Returns:
#             str: The prediction as a string representing a single integer[0, 3] for multiple choice tasks,
#                         or a string representing a comma separated list of integers for Ranking, Retrieval tasks,
#                         or a string representing a comma separated list of named entities for Named Entity Recognition tasks.
#                         or a string representing the (unconstrained) generated response for the generation tasks
#                         Please refer to parsers.py for more details on how these responses will be parsed by the evaluator.
#         """
#         possible_responses = [1, 2, 3, 4]

#         if is_multiple_choice:
#             # Randomly select one of the possible responses for multiple choice tasks
#             return str(random.choice(possible_responses))
#         else:
#             # For other tasks, shuffle the possible responses and return as a string
#             random.shuffle(possible_responses)
#             return str(possible_responses)
#             # Note: As this is dummy model, we are returning random responses for non-multiple choice tasks.
#             # For generation tasks, this should ideally return an unconstrained string.


class llama3_8b_FewShot(ShopBenchBaseModel):
    def __init__(self):
        random.seed(AICROWD_RUN_SEED)
        model_path = './models/Meta-Llama-3-8B-Instruct'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
        self.system_prompt =  "You are a helpful and multilingual online shopping assistant. You can understand and respond to user queries in English, German, Italian, French, Japanese, Spanish, Chinese. You are knowledgeable about various products. NOTE:ONLY OUTPUT THE ANSWER!!\n" 
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            # self.tokenizer.convert_tokens_to_ids("\\n"),
        ]
        self.load_rag_module(faiss_index_path="./models/index.ivf")

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
        
    def load_rag_module(self, faiss_index_path:str):
        # rag_module : embedding + faiss index + reranker
        self.embed_model = SentenceTransformer("./models/multilingual-e5-large-instruct", device='cpu')
        # self.reranker = FlagReranker('./models/bge-reranker-v2-m3', use_fp16=True, device='cuda:1')

        # few shot preprocess
        dim = 1024  # Embedding dimension for intfloat/multilingual-e5-large
        nlist = 1024 # Number of cluster centroids
        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = 3

        self.few_shot_example_text = []
        self.fewshot_embeddings = []
        with open('./models/large_sample_example.jsonl','r',encoding='utf8') as f:
            for i in f.readlines():
                passage = ''
                t_data = json.loads(i.strip())
                if "input" in t_data:
                    passage = t_data['instruction'] + t_data['input'] + '\nOutput:' + str( t_data['output']) + '\n'
                else:
                    passage = t_data['instruction'] + str(t_data['output']) + '\n'
                passage = passage.replace('\\n','\n')
                self.few_shot_example_text.append(passage)

        # preprocess train retrieve index and save trained index
        # self.fewshot_embeddings = self.embed_model.encode(self.few_shot_example_text, batch_size=128, show_progress_bar=True)
        # print(f'process few shot example embedding done! {len(self.few_shot_example_text)}')
        # self.index.train(self.fewshot_embeddings.astype(np.float32))
        # self.index.add(self.fewshot_embeddings.astype(np.float32))
        # faiss.write_index(self.index, "./models/index.ivf")

        self.index = faiss.read_index(faiss_index_path)
        self.metadata = [{"fewshot_examaple": fewshot_examaple} for fewshot_examaple in self.few_shot_example_text]

    def predict(self, prompt: str, is_multiple_choice: bool) -> str:
        faiss_retrieve_topk = 7
        faiss_score_filter = 0.88
        bge_rerank_topk = 6
        bge_score_filter = 0.6


        # faiss vector Retrieve smiliar few shot example
        task_description = "Given a online shopping user query, retrieve relevant Question-Answer that similar (type of task ,languages involved and product) to the query."
        query_text = ' ' + prompt
        query_embed = self.embed_model.encode([self.get_detailed_instruct(task_description, query_text)])[0]
        scores, indices = self.index.search(np.array([query_embed]).astype(np.float32), faiss_retrieve_topk)

        # process results
        exmaple_prompt = []
        for score, idx in zip(scores[0], indices[0]):
            if score>=faiss_score_filter and len(self.metadata[idx]["fewshot_examaple"])<=6000:
                fewshot_examaple = self.metadata[idx]["fewshot_examaple"]
                exmaple_prompt.append(fewshot_examaple)
        
        
        reranked_exmaple_prompt = exmaple_prompt if is_multiple_choice else exmaple_prompt[:4]

        # if len(exmaple_prompt)>0:
        #     print("before reranke:")
        #     print(exmaple_prompt[:4])
        #     # rerank the result
        #     retrank_task_description = "Given a online shopping user query, retrieve relevant Question-Answer that same (type of task ,languages involved) to the query."
        #     rerank_scores = self.reranker.compute_score(
        #         [[query_text,retrieved_fse] for retrieved_fse in exmaple_prompt],
        #         normalize=True,
        #         batch_size=32
        #     )
        #     reranked_exmaple_prompt = [
        #         exmaple_prompt[bge_rerank_topK_idx] 
        #             for bge_rerank_topK_idx in np.argsort(rerank_scores)[-bge_rerank_topk:] 
        #             if rerank_scores[bge_rerank_topK_idx]>=bge_score_filter
        #     ]
        #     reranked_exmaple_prompt = [t_prompt for t_prompt in reranked_exmaple_prompt if len(t_prompt)<=6000]
        #     print("reranked:")
        # else:
        #     reranked_exmaple_prompt = []

        if len(reranked_exmaple_prompt) > 0:
            prompt_example = self.system_prompt + '## Here are some similar questions and answers you can refer to:\n' 
            for i in reranked_exmaple_prompt:
                prompt_example += i+'\n'
            prompt_example += '## Now answer the Question:' + prompt
        else:
            prompt_example = self.system_prompt + '\n## Now answer the Question:' + prompt

        if is_multiple_choice:
            inputs = self.tokenizer.encode(prompt_example, add_special_tokens=False, return_tensors="pt").cuda()
            print("prompt token length: ",len(inputs[0]))
            generate_ids = self.model.generate(inputs, max_new_tokens=1, eos_token_id=self.terminators)
            result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            generation = result[len(prompt_example):]
        else:
            messages = [
                {"role": "system", "content": prompt_example[:len(self.system_prompt)]},
                {"role": "user", "content": prompt_example[len(self.system_prompt):]},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
            print("prompt token length :", len(input_ids[0]))
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=145,
                eos_token_id=self.terminators,
                do_sample=False,
            )[0][input_ids.shape[-1]:]
            generation = self.tokenizer.decode(outputs, skip_special_tokens=True)

        print(f'model generate answer : {generation}')
        return generation






class glm_9b_FewShot_vllm(ShopBenchBaseModel):
    """
    A dummy model implementation for ShopBench, illustrating how to handle both
    multiple choice and other types of tasks like Ranking, Retrieval, and Named Entity Recognition.
    This model uses a consistent random seed for reproducible results.
    """

    def __init__(self):
        """Initializes the model and sets the random seed for consistency."""
        random.seed(AICROWD_RUN_SEED)
        self.initialize_models()

    def initialize_models(self):
        # Initialize Meta Llama 3 - 8B Instruct Model
        self.model_name = "./models/glm-4-9b-chat"

        if not os.path.exists(self.model_name):
            raise Exception(
                f"""
            The evaluators expect the model weights to be checked into the repository,
            but we could not find the model weights at {self.model_name}
            
            Please follow the instructions in the docs below to download and check in the model weights.
                https://gitlab.aicrowd.com/aicrowd/challenges/amazon-kdd-cup-2024/amazon-kdd-cup-2024-starter-kit/-/blob/master/docs/download-baseline-model-weights.md
            
            """
            )

        # initialize the model with vllm
        self.system_prompt = "You are a helpful and multilingual online shopping assistant. You can understand and respond to user queries in English, German, Italian, French, Japanese, Spanish, Chinese. You are knowledgeable about various products. NOTE:ONLY OUTPUT THE ANSWER!!\n\n"

        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            trust_remote_code=True,
            dtype="bfloat16", # note: bfloat16 is not supported on nvidia-T4 GPUs
            enforce_eager=True,
            max_model_len=16*1024,
            # enable_chunked_prefill=True,
            # max_num_batched_tokens=8192
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.stop_token_ids  = [151329, 151336, 151338]

        self.faiss = self.load_rag_module(faiss_index_path="./models/large_index.ivf")
        self.faiss_retrieve_topk = 8
        self.faiss_score_filter = 0.88
        self.bge_rerank_topk = 6
        self.bge_score_filter = 0.6

        self.retrieve_task_description = "Given a online shopping user query, retrieve relevant Question-Answer that similar (type of task ,languages involved and product) to the query."

    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
        
    def load_rag_module(self, faiss_index_path: str):
        # rag_module : embedding + faiss index + reranker
        self.embed_model = SentenceTransformer("./models/multilingual-e5-large-instruct", device='cuda:1')
        # self.reranker = FlagReranker('./models/bge-reranker-v2-m3', use_fp16=True, device='cuda:1')

        # few shot preprocess
        self.few_shot_example_text = []
        with open('./models/large_sample_example.jsonl','r',encoding='utf8') as f:
            for i in f.readlines():
                passage = ''
                t_data = json.loads(i.strip())
                if "input" in t_data:
                    passage = t_data['instruction'] + t_data['input'] + '\nOutput:' + str( t_data['output']) + '\n'
                else:
                    passage = t_data['instruction'] + str(t_data['output']) + '\n'
                passage = passage.replace('\\n','\n')
                self.few_shot_example_text.append(passage)
        if os.path.exists(faiss_index_path):
            self.index = faiss.read_index(faiss_index_path)
        else:
            self.index = self.train_save_faiss_index(faiss_index_path)
        self.metadata = [{"fewshot_examaple": fewshot_examaple} for fewshot_examaple in self.few_shot_example_text]


    def train_save_faiss_index(self,
                               index_save_path: str = "./models/index.ivf",
                               dim: int = 1024,
                               nlist: int = 1024,
                               index_nprobe: int = 3):
        # preprocess train retrieve index and save trained index
        # dim : Embedding dimension for intfloat/multilingual-e5-large
        # nlist : Number of cluster centroids
        fewshot_embeddings = []
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = index_nprobe
        fewshot_embeddings = self.embed_model.encode(self.few_shot_example_text, batch_size=256+128, show_progress_bar=True)
        print(f'process few shot example embedding done! {len(self.few_shot_example_text)}')
        index.train(fewshot_embeddings.astype(np.float32))
        index.add(fewshot_embeddings.astype(np.float32))
        faiss.write_index(index, index_save_path)
        del fewshot_embeddings

        return index


    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_predict` function.

        Returns:
            int: The batch size, an integer between 1 and 16. This value indicates how many
                 queries should be processed together in a single batch. It can be dynamic
                 across different batch_predict calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE
        return self.batch_size


    def batch_predict(self, batch: Dict[str, Any], is_multiple_choice:bool) -> List[str]:
        prompts = batch["prompt"]
        
        # format prompts using the chat template
        formatted_prompts = self.format_prommpts(prompts, is_multiple_choice)
        # set max new tokens to be generated
        max_new_tokens = 156 
        
        if is_multiple_choice:
            max_new_tokens = 1 # For MCQ tasks, we only need to generate 1 token
        
        
        # Generate responses via vllm
        responses = self.llm.generate(
            formatted_prompts,
            sampling_params = vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                # top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                # top_k=1,
                temperature=0,  # randomness of the sampling
                seed=AICROWD_RUN_SEED, # Seed for reprodicibility
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=max_new_tokens,  # Maximum number of tokens to generate per output sequence.
                stop_token_ids=self.stop_token_ids # glm 4 stop token
            ),
            use_tqdm = False
        )
        # Aggregate answers into List[str]
        batch_response = []
        for response in responses:
            batch_response.append(response.outputs[0].text)        
            
        if is_multiple_choice:
            print("MCQ: ", batch_response)
        print(batch_response)
        return batch_response

    def format_prommpts(self, prompts:List[str], is_multiple_choice:bool):
        """
        Formats prompts using the chat_template of the model.
            
        Parameters:
        - queries (list of str): A list of queries to be formatted into prompts.
            
        """
        
        # 1. faiss index retrieve topK few shot example
        # 2. rerank few shot example
        # 3. select topK few shot example as prompt
        # 4.  [
        #       {"role":"system","content":self.system_prompt},
        #       {"role":"user","content": query + few shot exmaple}
        #       {"role":"assistant","content": model generate ... ...}
        #  ]

        # faiss vector Retrieve smiliar few shot example
        formatted_prompts = []
        for prompt in prompts:
            query_text = ' ' + prompt
            query_embed = self.embed_model.encode([self.get_detailed_instruct(self.retrieve_task_description, query_text)])[0]
            scores, indices = self.index.search(np.array([query_embed]).astype(np.float32), self.faiss_retrieve_topk)
            # process results
            few_shot_exmaple = []
            for score, retrieved_idx in zip(scores[0], indices[0]):
                if score>=self.faiss_score_filter and len(self.metadata[retrieved_idx]["fewshot_examaple"])<=6000:
                    fewshot_examaple = self.metadata[retrieved_idx]["fewshot_examaple"]
                    few_shot_exmaple.append(fewshot_examaple)
            
            
            reranked_exmaple_prompt = few_shot_exmaple   #  [:4] if  is_multiple_choice else few_shot_exmaple



            # rerank


            # if len(exmaple_prompt)>0:
            #     print("before reranke:")
            #     print(exmaple_prompt[:4])
            #     # rerank the result
            #     retrank_task_description = "Given a online shopping user query, retrieve relevant Question-Answer that same (type of task ,languages involved) to the query."
            #     rerank_scores = self.reranker.compute_score(
            #         [[query_text,retrieved_fse] for retrieved_fse in exmaple_prompt],
            #         normalize=True,
            #         batch_size=32
            #     )
            #     reranked_exmaple_prompt = [
            #         exmaple_prompt[bge_rerank_topK_idx] 
            #             for bge_rerank_topK_idx in np.argsort(rerank_scores)[-bge_rerank_topk:] 
            #             if rerank_scores[bge_rerank_topK_idx]>=bge_score_filter
            #     ]
            #     reranked_exmaple_prompt = [t_prompt for t_prompt in reranked_exmaple_prompt if len(t_prompt)<=6000]
            #     print("reranked:")
            # else:
            #     reranked_exmaple_prompt = []
            
            if len(reranked_exmaple_prompt) > 0:
                prompt_example = '## Here are some similar questions and answers you can refer to:\n' 
                for i in reranked_exmaple_prompt:
                    prompt_example += i+'\n'
                prompt_example += '## Now answer the Question:' + prompt
            else:
                prompt_example = '## Now answer the Question:' + prompt

                

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_example}
            ]
            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                # return_tensors="pt"
            )
            formatted_prompts.append(chat_prompt+'\n')

        return formatted_prompts

