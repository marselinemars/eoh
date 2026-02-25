import json
import re
import time
from ...llm.interface_LLM import InterfaceLLM
from .proposal_backends import get_proposal_backend

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
        self.proposal_mode = kwargs.get("proposal_mode", "eoh")
        self.proposal_backend = get_proposal_backend(self.proposal_mode)
        self.dx_call_mode = kwargs.get("dx_call_mode", "single")
        self.dx_observer_threshold_chars = kwargs.get("dx_observer_threshold_chars", 2200)
        self.last_proposal_info = None


        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM,llm_use_local,llm_local_url, self.debug_mode)

    def _safe_json_load(self, text):
        if not isinstance(text, str):
            return None
        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None

    def _need_observer_precall(self, dx_context):
        if self.proposal_mode != "dx":
            return False
        if self.dx_call_mode == "single":
            return False
        if not isinstance(dx_context, dict):
            return False

        if self.dx_call_mode == "observer_planner":
            return True

        # auto mode
        has_signal = bool(dx_context.get("parent_trace_summary")) or bool(dx_context.get("recent_history"))
        if not has_signal:
            return False
        try:
            context_chars = len(json.dumps(dx_context, ensure_ascii=True))
        except Exception:
            context_chars = 0
        return context_chars >= int(self.dx_observer_threshold_chars)

    def _observer_prompt(self, dx_context):
        return (
            "You are the Observer agent.\n"
            "Compress the diagnostic context into compact JSON for a planner.\n"
            "Output JSON only with keys: observation_summary, anomalies, regime_ranking.\n\n"
            "Input context:\n"
            + json.dumps(dx_context, ensure_ascii=True)
        )

    def _run_observer_precall(self, dx_context):
        info = {
            "observer_used": False,
            "observer_parse_error": None,
            "observer_summary": None,
            "observer_prompt": None,
            "observer_raw_response": None,
        }
        if not self._need_observer_precall(dx_context):
            return dx_context, info

        observer_prompt = self._observer_prompt(dx_context)
        observer_response = self.interface_llm.get_response(observer_prompt)
        observer_json = self._safe_json_load(observer_response)

        info["observer_used"] = True
        info["observer_prompt"] = observer_prompt
        info["observer_raw_response"] = observer_response

        if isinstance(observer_json, dict):
            compact = {
                "observation_summary": observer_json.get("observation_summary"),
                "anomalies": observer_json.get("anomalies"),
                "regime_ranking": observer_json.get("regime_ranking"),
            }
            info["observer_summary"] = compact
            new_context = dict(dx_context) if isinstance(dx_context, dict) else {}
            new_context["observer_summary"] = compact
            return new_context, info

        info["observer_parse_error"] = "invalid_observer_json"
        # Keep original context if observer output is invalid.
        return dx_context, info

    def _prepare_prompt(self, operator, base_prompt, dx_context):
        effective_context, observer_info = self._run_observer_precall(dx_context)
        prompt_content = self.proposal_backend.build_prompt(operator, base_prompt, effective_context)
        return prompt_content, observer_info

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


    def _legacy_extract(self, response):
        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        if len(algorithm) == 0 or len(code) == 0:
            return None
        return code[0], algorithm[0]

    def _looks_like_complete_return(self, code):
        if not isinstance(code, str):
            return False
        # JSON-path proposals usually return complete function text.
        # Legacy regex path often truncates at keyword `return`.
        return bool(re.search(r"return\s+[^\s].*", code))


    def _get_alg(self,prompt_content, observer_info=None):

        response = self.interface_llm.get_response(prompt_content)
        proposal_meta = {
            "proposal_mode": self.proposal_mode,
            "prompt": prompt_content,
            "raw_response": response,
            "used_json": False,
            "parse_error": None,
            "parsed_json": None,
            "fallback_used": False,
            "retry_count": 0,
            "observer_used": False,
            "observer_parse_error": None,
            "observer_summary": None,
            "observer_prompt": None,
            "observer_raw_response": None,
        }
        if isinstance(observer_info, dict):
            proposal_meta.update(observer_info)

        parsed = None
        if self.proposal_mode == "dx":
            code_json, algorithm_json, parse_meta = self.proposal_backend.parse_response(response)
            proposal_meta["used_json"] = bool(parse_meta.get("used_json"))
            proposal_meta["parse_error"] = parse_meta.get("parse_error")
            proposal_meta["parsed_json"] = parse_meta.get("parsed_json")
            if code_json is not None and algorithm_json is not None:
                parsed = (code_json, algorithm_json)
            else:
                proposal_meta["fallback_used"] = True
                parsed = self._legacy_extract(response)
        else:
            parsed = self._legacy_extract(response)

        n_retry = 1
        while parsed is None:
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)
            proposal_meta["raw_response"] = response
            proposal_meta["retry_count"] = n_retry

            if self.proposal_mode == "dx":
                code_json, algorithm_json, parse_meta = self.proposal_backend.parse_response(response)
                proposal_meta["used_json"] = bool(parse_meta.get("used_json"))
                proposal_meta["parse_error"] = parse_meta.get("parse_error")
                proposal_meta["parsed_json"] = parse_meta.get("parsed_json")
                if code_json is not None and algorithm_json is not None:
                    parsed = (code_json, algorithm_json)
                else:
                    proposal_meta["fallback_used"] = True
                    parsed = self._legacy_extract(response)
            else:
                parsed = self._legacy_extract(response)
                
            if n_retry > 3:
                break
            n_retry +=1

        if parsed is None:
            # Keep old behavior: this will raise and be handled by caller.
            self.last_proposal_info = proposal_meta
            raise ValueError("Could not parse algorithm or code from LLM response.")
        code, algorithm = parsed

        # Keep legacy behavior for regex-truncated code; avoid corrupting
        # JSON-path code that already contains a full return expression.
        if proposal_meta.get("used_json") and self._looks_like_complete_return(code):
            code_all = code
        else:
            code_all = code + " " + ", ".join(s for s in self.prompt_func_outputs)

        self.last_proposal_info = proposal_meta
        return [code_all, algorithm, proposal_meta]


    def i1(self, dx_context=None):

        base_prompt = self.get_prompt_i1()
        prompt_content, observer_info = self._prepare_prompt("i1", base_prompt, dx_context or {"parents": None})

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm, proposal_info] = self._get_alg(prompt_content, observer_info=observer_info)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm, proposal_info]
    
    def e1(self,parents, dx_context=None):
      
        base_prompt = self.get_prompt_e1(parents)
        prompt_content, observer_info = self._prepare_prompt("e1", base_prompt, dx_context or {"parents": parents})

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm, proposal_info] = self._get_alg(prompt_content, observer_info=observer_info)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm, proposal_info]
    
    def e2(self,parents, dx_context=None):
      
        base_prompt = self.get_prompt_e2(parents)
        prompt_content, observer_info = self._prepare_prompt("e2", base_prompt, dx_context or {"parents": parents})

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm, proposal_info] = self._get_alg(prompt_content, observer_info=observer_info)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm, proposal_info]
    
    def m1(self,parents, dx_context=None):
      
        base_prompt = self.get_prompt_m1(parents)
        prompt_content, observer_info = self._prepare_prompt("m1", base_prompt, dx_context or {"parents": [parents]})

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm, proposal_info] = self._get_alg(prompt_content, observer_info=observer_info)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm, proposal_info]
    
    def m2(self,parents, dx_context=None):
      
        base_prompt = self.get_prompt_m2(parents)
        prompt_content, observer_info = self._prepare_prompt("m2", base_prompt, dx_context or {"parents": [parents]})

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm, proposal_info] = self._get_alg(prompt_content, observer_info=observer_info)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm, proposal_info]
    
    def m3(self,parents, dx_context=None):
      
        base_prompt = self.get_prompt_m3(parents)
        prompt_content, observer_info = self._prepare_prompt("m3", base_prompt, dx_context or {"parents": [parents]})

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m3 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm, proposal_info] = self._get_alg(prompt_content, observer_info=observer_info)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm, proposal_info]
