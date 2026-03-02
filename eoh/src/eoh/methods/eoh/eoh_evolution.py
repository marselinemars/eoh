import re
import time
import json
import os
import numpy as np
from ...llm.interface_LLM import InterfaceLLM

class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs):

        # set prompt interface
        #getprompts = GetPrompts()
        self.prompt_task         = prompts.get_task()
        self.prompt_func_name    = prompts.get_func_name()
        self.prompt_func_inputs  = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf    = prompts.get_inout_inf()
        self.prompt_other_inf    = prompts.get_other_inf()
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode # close prompt checking

        self.log_llm_interactions = bool(kwargs.get("log_llm_interactions", False))
        self.llm_log_dir = kwargs.get("llm_log_dir", None)
        self._llm_log_counter = 0
        if self.log_llm_interactions and self.llm_log_dir:
            os.makedirs(self.llm_log_dir, exist_ok=True)

        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM,llm_use_local,llm_local_url, self.debug_mode)

    def _save_llm_interaction(self, prompt_text, raw_response, parsed_algorithm, parsed_code, generation_idx, operator_name):
        if (not self.log_llm_interactions) or (not self.llm_log_dir):
            return
        generation_id = 0 if generation_idx is None else int(generation_idx)
        op_name = "unknown" if operator_name is None else str(operator_name)
        base_name = f"generation_{generation_id}_operator_{op_name}"
        file_path = os.path.join(self.llm_log_dir, f"{base_name}.json")
        while os.path.exists(file_path):
            self._llm_log_counter += 1
            file_path = os.path.join(self.llm_log_dir, f"{base_name}_{self._llm_log_counter}.json")

        payload = {
            "prompt": prompt_text,
            "response": raw_response,
            "parsed_algorithm": parsed_algorithm,
            "parsed_code": parsed_code,
        }
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=True)

    def _extract_algorithm(self, response: str) -> str:
        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm) > 0:
            return algorithm[0].strip()
        for line in response.splitlines():
            line = line.strip()
            if line:
                return line
        return "LLM-generated score heuristic"

    def _extract_function_block(self, text: str) -> str:
        lines = text.splitlines()
        func_pat = re.compile(r"^\s*def\s+" + re.escape(self.prompt_func_name) + r"\s*\(")
        start = None
        for i, line in enumerate(lines):
            if func_pat.match(line):
                start = i
                break
        if start is None:
            return ""

        import_lines = []
        for line in lines[:start]:
            s = line.strip()
            if s.startswith("import ") or s.startswith("from "):
                import_lines.append(s)

        block = [lines[start]]
        for j in range(start + 1, len(lines)):
            line = lines[j]
            if line.strip() == "":
                block.append(line)
                continue
            if line.startswith(" ") or line.startswith("\t"):
                block.append(line)
                continue
            break

        code = "\n".join(import_lines + block).strip() + "\n"
        if "import numpy as np" not in code:
            code = "import numpy as np\n" + code
        return code

    def _validate_code(self, code: str) -> bool:
        try:
            compiled = compile(code, "<llm_code>", "exec")
            ns = {}
            exec(compiled, ns)
            fn = ns.get(self.prompt_func_name)
            if fn is None or not callable(fn):
                return False
            bins = np.array([10.0, 20.0], dtype=float)
            out = fn(5.0, bins)
            arr = np.asarray(out, dtype=float)
            return arr.shape == bins.shape
        except Exception:
            return False

    def _extract_code(self, response: str) -> str:
        candidates = re.findall(r"```(?:python)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
        if "import " in response:
            candidates.append(response[response.find("import "):])
        if "def " in response:
            candidates.append(response[response.find("def "):])
        candidates.append(response)

        for cand in candidates:
            code = self._extract_function_block(cand)
            if not code:
                continue
            if self._validate_code(code):
                return code

        return (
            "import numpy as np\n"
            "def score(item, bins):\n"
            "    return -bins\n"
        )

    def get_prompt_i1(self):
        
        prompt_content = self.prompt_task+"\n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content

        
    def get_prompt_e1(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+\
"Please help me create a new algorithm that has a totally different form from the given ones. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    def get_prompt_e2(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+\
"Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n"\
"Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    def get_prompt_m1(self,indiv1):
        prompt_content = self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    def get_prompt_m2(self,indiv1):
        prompt_content = self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    def get_prompt_m3(self,indiv1):
        prompt_content = "First, you need to identify the main components in the function below. \
Next, analyze whether any of these components can be overfit to the in-distribution instances. \
Then, based on your analysis, simplify the components to enhance the generalization to potential out-of-distribution instances. \
Finally, provide the revised code, keeping the function name, inputs, and outputs unchanged. \n"+indiv1['code']+"\n"\
+self.prompt_inout_inf+"\n"+"Do not give additional explanations."
        return prompt_content


    def _get_alg(self,prompt_content, generation_idx=None, operator_name=None):
        response = ""
        algorithm = "LLM-generated score heuristic"
        code_all = ""

        n_retry = 0
        while n_retry <= 3:
            response = self.interface_llm.get_response(prompt_content)
            algorithm = self._extract_algorithm(response)
            code_all = self._extract_code(response)
            if self._validate_code(code_all):
                break
            n_retry += 1

        self._save_llm_interaction(
            prompt_text=prompt_content,
            raw_response=response,
            parsed_algorithm=algorithm,
            parsed_code=code_all,
            generation_idx=generation_idx,
            operator_name=operator_name,
        )

        return [code_all, algorithm]


    def i1(self, generation_idx=None, operator_name="i1"):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content, generation_idx=generation_idx, operator_name=operator_name)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def e1(self,parents, generation_idx=None, operator_name="e1"):
      
        prompt_content = self.get_prompt_e1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content, generation_idx=generation_idx, operator_name=operator_name)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def e2(self,parents, generation_idx=None, operator_name="e2"):
      
        prompt_content = self.get_prompt_e2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content, generation_idx=generation_idx, operator_name=operator_name)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m1(self,parents, generation_idx=None, operator_name="m1"):
      
        prompt_content = self.get_prompt_m1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content, generation_idx=generation_idx, operator_name=operator_name)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m2(self,parents, generation_idx=None, operator_name="m2"):
      
        prompt_content = self.get_prompt_m2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content, generation_idx=generation_idx, operator_name=operator_name)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m3(self,parents, generation_idx=None, operator_name="m3"):
      
        prompt_content = self.get_prompt_m3(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m3 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content, generation_idx=generation_idx, operator_name=operator_name)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
