import re
import time
import ast
import os
import json
import types
import numpy as np
from pathlib import Path
from datetime import datetime
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


        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM,llm_use_local,llm_local_url, self.debug_mode)
        self.max_parse_retries = int(os.getenv("EOH_PARSE_RETRIES", "6"))
        self.log_parse_events = os.getenv("EOH_LOG_PARSE_EVENTS", "0") == "1"
        self.llm_io_dir = Path(os.getenv("EOH_LLM_IO_DIR", "./results/llm_io"))
        self.parse_log_path = self.llm_io_dir / "parse_events.jsonl"
        self.active_prompt_modifiers = []
        if self.log_parse_events:
            self.llm_io_dir.mkdir(parents=True, exist_ok=True)

    def set_prompt_modifiers(self, modifiers):
        if not isinstance(modifiers, list):
            self.active_prompt_modifiers = []
            return
        cleaned = []
        for item in modifiers:
            text = str(item).strip()
            if text:
                cleaned.append(text)
        self.active_prompt_modifiers = cleaned[:4]

    def _append_additional_constraints(self, prompt_content):
        if len(self.active_prompt_modifiers) == 0:
            return prompt_content
        extra = "\nADDITIONAL CONSTRAINTS FOR THIS GENERATION:\n"
        for modifier in self.active_prompt_modifiers:
            extra += f"- {modifier}\n"
        return prompt_content + extra

    def _strict_output_rules(self):
        return (
            "STRICT OUTPUT FORMAT (MANDATORY):\n"
            "1) First line: one sentence wrapped in braces like {your sentence}.\n"
            "2) Then output ONLY Python code (no markdown fences).\n"
            "3) Code must include: import numpy as np\n"
            f"4) Code must define exactly one function named {self.prompt_func_name}.\n"
            f"5) Function inputs must be exactly: ({', '.join(self.prompt_func_inputs)}).\n"
            f"6) Function must return: {', '.join(self.prompt_func_outputs)}.\n"
            "7) Do NOT output 'Thinking Process', analysis, explanations, bullet points, or prose.\n"
            "8) Do NOT output anything before the brace line or after the Python code.\n"
        )

    def _log_parse_event(self, event, **fields):
        if not self.log_parse_events:
            return
        record = {
            "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "event": event,
            "request_id": getattr(self.interface_llm, "last_request_id", None),
            "func_name": self.prompt_func_name,
        }
        record.update(fields)
        with self.parse_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def get_prompt_i1(self):
        
        prompt_content = self.prompt_task+"\n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+"The new heuristic must produce materially different bin rankings across some feasible-bin cases; avoid trivial rescaling or cosmetic edits. "\
+self.prompt_other_inf+"\n"+self._strict_output_rules()
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
+"The new heuristic must change ranking behavior substantially relative to the provided algorithms. Do not output a cosmetic rewrite, simple threshold mask, or coefficient rescale. "\
+self.prompt_other_inf+"\n"+self._strict_output_rules()
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
+"The variant must preserve only the useful motif, while changing bin-ranking behavior on some feasible-bin cases. Reject trivial restatements of a parent. "\
+self.prompt_other_inf+"\n"+self._strict_output_rules()
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
+"The modified algorithm must alter ranking behavior on some feasible-bin cases. Do not just rewrite variable names or apply a cosmetic penalty. "\
+self.prompt_other_inf+"\n"+self._strict_output_rules()
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
+"A pure coefficient rescale that preserves the same ranking will be rejected. Adjust parameters only if the resulting heuristic changes bin choices on some cases. "\
+self.prompt_other_inf+"\n"+self._strict_output_rules()
        return prompt_content
    
    def get_prompt_m3(self,indiv1):
        prompt_content = "First, you need to identify the main components in the function below. \
Next, analyze whether any of these components can be overfit to the in-distribution instances. \
Then, based on your analysis, simplify the components to enhance the generalization to potential out-of-distribution instances. \
Finally, provide the revised code, keeping the function name, inputs, and outputs unchanged. The simplification must still change ranking behavior on some feasible-bin cases; do not return a cosmetic rewrite. \n"+indiv1['code']+"\n"\
+self.prompt_inout_inf+"\n"+self._strict_output_rules()
        return prompt_content


    def _normalize_text(self, text):
        if text is None:
            return ""
        text = text.replace("\r\n", "\n")
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        return text

    def _extract_algorithm(self, response):
        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm) > 0:
            return algorithm[0].strip()
        for splitter in ["```", "def ", "import ", "from "]:
            if splitter in response:
                return response.split(splitter)[0].strip()
        return "Generated heuristic"

    def _extract_code_block(self, response):
        func_pat = rf"def\s+{re.escape(self.prompt_func_name)}\s*\("
        code_blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        for block in code_blocks:
            if re.search(func_pat, block):
                return block.strip()
        if len(code_blocks) > 0:
            return code_blocks[0].strip()

        m_def = re.search(rf"(def\s+{re.escape(self.prompt_func_name)}\s*\(.*)", response, re.DOTALL)
        if m_def:
            code = m_def.group(1)
            code = code.split("```")[0]
            return code.strip()

        m_import_def = re.search(rf"(import[\s\S]*?def\s+{re.escape(self.prompt_func_name)}\s*\(.*)", response, re.DOTALL)
        if m_import_def:
            code = m_import_def.group(1)
            code = code.split("```")[0]
            return code.strip()
        return None

    def _build_valid_code(self, code_candidate):
        code = code_candidate.strip()
        if f"def {self.prompt_func_name}" not in code:
            raise RuntimeError("Missing required function definition.")
        if "import numpy as np" not in code:
            code = "import numpy as np\n\n" + code
        tree = ast.parse(code)
        banned_nodes = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        for node in ast.walk(tree):
            if isinstance(node, banned_nodes):
                raise RuntimeError("score() must use vectorized numpy operations; Python loops/comprehensions are not allowed.")
            if isinstance(node, ast.Call):
                func = node.func
                attr_name = None
                if isinstance(func, ast.Attribute):
                    attr_name = func.attr
                elif isinstance(func, ast.Name):
                    attr_name = func.id
                if attr_name in {"sort", "argsort", "lexsort", "apply_along_axis", "vectorize", "frompyfunc"}:
                    raise RuntimeError(f"score() uses {attr_name}(), which is too expensive for repeated online evaluation.")
        self._validate_runtime_contract(code)
        return code

    def _validate_runtime_contract(self, code):
        if self.prompt_func_name != "score" or self.prompt_func_inputs != ["item", "bins"]:
            return
        module = types.ModuleType("candidate_heuristic")
        exec(code, module.__dict__)
        if not hasattr(module, self.prompt_func_name):
            raise RuntimeError("Generated code does not expose the required score function.")
        probe_cases = [
            (3, np.asarray([3.0, 5.0, 7.0], dtype=float)),
            (6, np.asarray([6.0, 8.0, 10.0, 12.0, 15.0], dtype=float)),
            (4, np.linspace(4.0, 32.0, num=17, dtype=float)),
            (9, np.linspace(9.0, 5008.0, num=5000, dtype=float)),
        ]
        saw_nontrivial_variation = False
        saw_length_variation = False
        for item, bins in probe_cases:
            t0 = time.perf_counter()
            try:
                raw = getattr(module, self.prompt_func_name)(item, bins.copy())
            except Exception as exc:
                raise RuntimeError(f"score() execution failed on probe len={len(bins)}: {exc}") from exc
            elapsed = time.perf_counter() - t0
            try:
                scores = np.asarray(raw, dtype=float).reshape(-1)
            except Exception as exc:
                raise RuntimeError(f"score() output is not a numeric vector: {exc}") from exc
            if len(scores) != len(bins):
                raise RuntimeError("score() must return one numeric score per feasible bin.")
            if not np.all(np.isfinite(scores)):
                raise RuntimeError("score() output contains NaN or infinity.")
            if len(bins) >= 5000 and elapsed > 0.02:
                raise RuntimeError("score() is too slow on probe cases; use vectorized numpy operations.")
            if np.max(scores) > np.min(scores):
                saw_nontrivial_variation = True
            if len(np.unique(np.round(scores, 12))) > 1:
                saw_length_variation = True
        large_bins = np.linspace(9.0, 5008.0, num=5000, dtype=float)
        t0 = time.perf_counter()
        for _ in range(32):
            raw = getattr(module, self.prompt_func_name)(9, large_bins.copy())
            scores = np.asarray(raw, dtype=float).reshape(-1)
            if len(scores) != len(large_bins) or not np.all(np.isfinite(scores)):
                raise RuntimeError("score() failed repeated large-array probe.")
        repeated_elapsed = time.perf_counter() - t0
        if repeated_elapsed > 0.15:
            raise RuntimeError("score() is too slow for repeated online use; repeated large-array probe exceeded runtime budget.")
        if not saw_nontrivial_variation:
            raise RuntimeError("score() output is constant on probe cases; heuristic is too weak/degenerate.")
        if not saw_length_variation:
            raise RuntimeError("score() does not meaningfully vary across feasible bins on probe cases.")

    def _fallback_code(self):
        if self.prompt_func_name == "score" and self.prompt_func_inputs == ["item", "bins"]:
            return (
                "import numpy as np\n\n"
                "def score(item, bins):\n"
                "    # Simple stable fallback: favor tighter fit.\n"
                "    return -np.abs(bins - item)\n"
            )
        args = ", ".join(self.prompt_func_inputs)
        if len(self.prompt_func_outputs) == 1:
            ret = "0.0"
        else:
            ret = "(" + ", ".join(["0.0"] * len(self.prompt_func_outputs)) + ")"
        return (
            "import numpy as np\n\n"
            f"def {self.prompt_func_name}({args}):\n"
            f"    return {ret}\n"
        )

    def _get_alg(self,prompt_content):
        last_err = None
        for i in range(self.max_parse_retries):
            response = self.interface_llm.get_response(prompt_content)
            response = self._normalize_text(response)
            algorithm = self._extract_algorithm(response)
            code_candidate = self._extract_code_block(response)
            if code_candidate is None:
                last_err = RuntimeError("No code block detected in response.")
                self._log_parse_event(
                    "parse_fail",
                    attempt=i + 1,
                    error=str(last_err),
                    response_preview=response[:400],
                )
                continue
            try:
                code_all = self._build_valid_code(code_candidate)
                self._log_parse_event(
                    "parse_success",
                    attempt=i + 1,
                    code_chars=len(code_all),
                    algorithm=algorithm,
                )
                return [code_all, algorithm]
            except Exception as exc:
                last_err = exc
                self._log_parse_event(
                    "parse_fail",
                    attempt=i + 1,
                    error=str(exc),
                    code_preview=code_candidate[:400],
                    response_preview=response[:400],
                )
                if self.debug_mode:
                    print(f"Parse/build failure {i+1}/{self.max_parse_retries}: {exc}")
                time.sleep(0.5)

        if self.debug_mode:
            print(f"Falling back to deterministic code due to parse errors: {last_err}")
        self._log_parse_event("parse_fallback", error=str(last_err))
        return [self._fallback_code(), "Fallback valid heuristic"]


    def i1(self):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def generate_from_prompt(self, prompt_content):
        prompt_content = self._append_additional_constraints(prompt_content)
        return self._get_alg(prompt_content)
    
    def e1(self,parents):
      
        prompt_content = self.get_prompt_e1(parents)
        prompt_content = self._append_additional_constraints(prompt_content)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def e2(self,parents):
      
        prompt_content = self.get_prompt_e2(parents)
        prompt_content = self._append_additional_constraints(prompt_content)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m1(self,parents):
      
        prompt_content = self.get_prompt_m1(parents)
        prompt_content = self._append_additional_constraints(prompt_content)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m2(self,parents):
      
        prompt_content = self.get_prompt_m2(parents)
        prompt_content = self._append_additional_constraints(prompt_content)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m3(self,parents):
      
        prompt_content = self.get_prompt_m3(parents)
        prompt_content = self._append_additional_constraints(prompt_content)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m3 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
