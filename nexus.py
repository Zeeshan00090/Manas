from sympy.polys.polyconfig import query
import threading
import time
import logging
import gc
import numpy as np
import queue
import math
import requests
import json
import os
import subprocess
import wikipedia
from duckduckgo_search import DDGS

from typing import Dict, Any, List

# --- Local Modules ---
# Import the massive monolith modules (1-11)
from Manas import (
    GoalNode, GoalTree, HierarchicalPlanner,
    RLAgentEnvironment, QActionPolicy,
    ConceptOntology, SymbolGrounder,
    MetacognitiveSelfModel, CrossDomainAdapter,
    # Module 4
    System_Load_Balancer, ComputeTier,
    # Module 6
    Temporal_Graph_Storage, NodeKind,
    # Module 3
    Hardware_Entropy_Harvester,
    # Data structures
    ManasSystemState
)

from understanding_extractor import UnderstandingExtractor
from auto_self_trainer import AutoSelfTrainer

log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, "manas_nexus_system.log")

# Truncate old stale log file on startup
try:
    with open(log_file, 'w', encoding='utf-8') as f:
        f.truncate(0)
except Exception as e:
    print(f"Failed to clear log file: {e}")

# Explicitly configure root logger handlers to bypass basicConfig block
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)

log_formatter = logging.Formatter('%(asctime)s - [MANAS AGI] - %(message)s')

stream_h = logging.StreamHandler()
stream_h.setFormatter(log_formatter)
root_logger.addHandler(stream_h)

try:
    file_h = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_h.setFormatter(log_formatter)
    root_logger.addHandler(file_h)
except Exception as e:
    print(f"Failed to add file log handler: {e}")

# ==========================================
# OLLAMA LLM INTEGRATION
# ==========================================
class OllamaEngine:
    def __init__(self, model="dolphin-llama3:latest", host="http://localhost:11434"):
        self.model = model
        self.host = host
        
    def generate(self, prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
        try:
            response = requests.post(f"{self.host}/api/generate", json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }, timeout=45)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logging.error(f"Ollama error: {response.text}")
                return ""
        except Exception as e:
            logging.error(f"Ollama connection failed: {e}")
            return ""

# ==========================================
# LIQUID ZULIC KAN LAYER (FIXED MATH)
# ==========================================
class LiquidZULICKANLayer:
    def __init__(self, input_size, output_size, layer_id=0, tau=0.5):
        self.layer_id = layer_id
        self.tau = tau
        self.state = np.zeros(output_size)
        self.w_base = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.w_sin = np.random.randn(input_size, output_size) * 0.1

    def forward(self, x, dt=0.1):
        # Handle both 1D and 2D inputs
        edge = np.dot(x, self.w_base) + np.dot(np.sin(x), self.w_sin)
        
        # If x is batched (2D), we take the mean edge to update the global layer state
        if x.ndim == 2:
            self.state += (-(self.state / self.tau) + np.mean(edge, axis=0)) * dt
        else:
            self.state += (-(self.state / self.tau) + edge) * dt
            
        return np.maximum(0, self.state if x.ndim == 1 else edge + self.state)

    def normalize(self, x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    def liquid_zulic_update(self, pos_x, neg_x, flow_rate=0.01, viscosity=0.005):
        pos_out = self.forward(pos_x)
        neg_out = self.forward(neg_x)
        
        # Support batch matrix multiplication mathematically
        if pos_x.ndim == 2:
            batch_size = pos_x.shape[0]
            grad_base = (pos_x.T @ pos_out - neg_x.T @ neg_out) / batch_size
            grad_sin = (np.sin(pos_x).T @ pos_out - np.sin(neg_x).T @ neg_out) / batch_size
        else:
            grad_base = np.outer(pos_x, pos_out) - np.outer(neg_x, neg_out)
            grad_sin = np.outer(np.sin(pos_x), pos_out) - np.outer(np.sin(neg_x), neg_out)
            
        self.w_base += flow_rate * grad_base - (viscosity * self.w_base)
        self.w_sin += flow_rate * grad_sin - (viscosity * self.w_sin)
        return self.normalize(pos_out), self.normalize(neg_out)


# ==========================================
# MANAS GRAND UNIFIED NEXUS (v9.0)
# ==========================================
class ManasGrandUnifiedNexus:
    def __init__(self):
        logging.info("Initializing Manas AGI 9.0 Grand Unified Engine...")
        
        self.is_running = True
        self.input_queue = queue.Queue()
        self.training_queue = queue.Queue()
        self.step_counter = 0
        
        # --- AGI Subsystems ---
        self.llm = OllamaEngine()
        self.planner = HierarchicalPlanner(llm_generate_fn=self.llm.generate)
        self.rl_env = RLAgentEnvironment()
        self.rl_policy = QActionPolicy()
        self.ontology = ConceptOntology()
        self.grounder = SymbolGrounder(self.ontology)
        self.metacognition = MetacognitiveSelfModel()
        self.cross_domain = CrossDomainAdapter(self.ontology)
        
        # --- Physical/Memory Subsystems ---
        self.balancer = System_Load_Balancer()
        self.graph = Temporal_Graph_Storage(embedding_dim=128, persist_dir=os.path.join(log_dir, "graph_db"))
        self.entropy = Hardware_Entropy_Harvester()
        
        # --- Liquid Core ---
        self.liquid_core = [LiquidZULICKANLayer(128, 128, i) for i in range(3)]
        self.liquid_pos_buffer = []
        self.liquid_neg_buffer = []
        
        # --- External Extractors ---
        self.understanding = UnderstandingExtractor()
        
        # Connect AutoSelfTrainer with correct signatures and real thermal guard
        from thermal_guard import get_global_guard
        tg = get_global_guard()
        
        def get_trainer_resources():
            status_str = tg.status_string()
            status_dict = tg.get_status()
            bio_status = "CRITICAL" if status_dict["is_cooling"] else "OK"
            return status_str, bio_status

        def get_curiosity_topic_and_questions():
            gaps = self.understanding.get_top_gaps(1)
            if gaps:
                return gaps[0].topic, gaps[0].questions
            return None, []

        def get_causal_facts():
            return [f for f in self.understanding._fact_store if f.fact_type in ["CAUSE", "EFFECT", "CONDITIONAL"]]

        self.self_trainer = AutoSelfTrainer(
            llm_generate_fn=self.llm.generate,
            self_eval_fn=self.self_evaluate,
            get_resource_fn=get_trainer_resources,
            get_curiosity_fn=get_curiosity_topic_and_questions,
            get_causal_facts_fn=get_causal_facts,
            ft_data_dir=os.path.join(log_dir, "manas_training_data"),
            solve_code_fn=self.solve_via_code
        )
        
        self.conversation_history = []
        self.last_user_input_time = time.time()
        self.learned_topics = set()
        
        # Seed initial knowledge to stimulate curiosity and facts
        initial_seeds = [
            "Artificial General Intelligence is a type of intelligence that can solve any cognitive task.",
            "Artificial General Intelligence requires Quantum Physics to function.",
            "Artificial General Intelligence can utilize Dynamic Programming.",
            "Quantum Physics is a branch of physics that studies subatomic particles.",
            "Quantum Physics relates to quantum computers.",
            "Dynamic Programming is an optimization technique that solves problems recursively.",
            "Dynamic Programming optimizes Reinforcement Learning.",
            "Reinforcement Learning is a learning paradigm based on reward maximization.",
            "Reinforcement Learning is used to train Artificial General Intelligence."
        ]
        for seed in initial_seeds:
            try:
                knowledge = self.understanding.extract(seed, source="seed_generator")
                for fact in knowledge.get("facts", []):
                    self.ontology.add_relation(fact.subject, fact.predicate, fact.obj)
            except Exception as e:
                logging.error(f"Error seeding knowledge: {e}")
        
        # Start background threads
        self.threads = [
            threading.Thread(target=self._logic_loop, name="Logic-Thread", daemon=True),
            threading.Thread(target=self._trainer_loop, name="Trainer-Thread", daemon=True),
            threading.Thread(target=self._dream_loop, name="Dreamer-Thread", daemon=True),
            threading.Thread(
                target=self.self_trainer.autonomous_loop,
                args=(lambda: self.last_user_input_time,),
                name="Self-Play-Trainer-Thread",
                daemon=True
            )
        ]
        for t in self.threads:
            t.start()
            
        logging.info("Manas 9.0 initialized and deeply integrated.")

    def _execute_sandbox(self, code: str) -> tuple:
        """
        Executes code in a subprocess sandbox with CWD set strictly to the `./ai_agents/` directory.
        """
        try:
            sandbox_dir = "/Users/zeeshanrahman/Manas/ai_agents"
            os.makedirs(sandbox_dir, exist_ok=True)
            result = subprocess.run(
                ["python3"],
                input=code,
                capture_output=True,
                text=True,
                timeout=15,
                cwd=sandbox_dir
            )
            success = (result.returncode == 0)
            return success, result.stdout, result.stderr
        except subprocess.TimeoutExpired as e:
            return False, "", f"TimeoutExpired: Execution timed out after 15 seconds.\nStdout: {e.stdout}\nStderr: {e.stderr}"
        except Exception as e:
            return False, "", f"ExecutionFailed: {str(e)}"

    def _is_code_safe(self, code: str) -> tuple:
        """
        Safety filter to check if code attempts to escape the './ai_agents/' sandbox.
        """
        forbidden_keywords = ["os.system", "shutil", "subprocess.run", "subprocess.Popen", "eval(", "exec("]
        for keyword in forbidden_keywords:
            if keyword in code:
                return False, f"SafetyViolation: Using forbidden keyword/module '{keyword}'. You are not allowed to execute shell commands or run subprocesses."

        # Traversal check
        if ".." in code:
            return False, "SafetyViolation: Parent directory traversal ('..') is strictly forbidden to protect the parent codebase."

        # Absolute path check: Block absolute paths that might escape the sandbox
        import re
        strings = re.findall(r"'(.*?)'|\"(.*?)\"", code)
        for s_tuple in strings:
            s = s_tuple[0] or s_tuple[1]
            if s.startswith("/") and not s.startswith("/Users/zeeshanrahman/Manas/ai_agents"):
                return False, f"SafetyViolation: Absolute path access '{s}' is forbidden. All operations must be relative to the './ai_agents/' directory."
                
        return True, ""

    def solve_via_code(self, query: str) -> tuple:
        """
        Runs the Infinite Fix Loop:
        Draft Code -> Safety Check -> Run Sandbox -> Check Error -> If Error/Violation, Self-Correct -> Loop (max 4 retries)
        Returns (stdout_result_or_error, final_code, success_boolean)
        """
        # 1. Draft initial Python code
        code_prompt = f"""<|im_start|>system
You are Manas AGI, an advanced AI engine. Your goal is to write a complete, self-contained Python 3 script to solve or answer this query: "{query}"

Rules:
1. Output ONLY valid, executable Python code. Do not wrap it in markdown code blocks like ```python. Just output the raw code.
2. The script must execute, compute the final answer, and PRINT the final result to standard output.
3. You can import standard libraries (e.g. math, json, urllib.request, time, datetime) or pre-installed libraries (numpy, requests, duckduckgo_search, wikipedia).
4. If you need information from the internet, you can use `duckduckgo_search` or `wikipedia` package to fetch it inside the code, parse it, and print the answer.
5. The working directory of the sandbox is `/Users/zeeshanrahman/Manas/ai_agents`. Any files you create without a path prefix (e.g. `open('hello.txt')`) will automatically be saved inside this directory. Do NOT use absolute paths.
6. Do not include any conversational explanation. Only code.
<|im_end|>
<|im_start|>user
Write the code to solve: {query}<|im_end|>
<|im_start|>assistant
"""
        code = self.llm.generate(code_prompt, max_tokens=1000, temperature=0.2).strip()
        # Clean markdown formatting and conversational wrapper robustly
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        code = code.strip()

        max_retries = 4
        retry_count = 0
        success = False
        output = ""
        error_log = ""

        logging.info(f"Drafted initial code:\n{code}")

        while retry_count <= max_retries:
            # Enforce AI Agents safety boundary checks
            is_safe, safety_error = self._is_code_safe(code)
            if not is_safe:
                success = False
                error_log = safety_error
            else:
                success, output, error_log = self._execute_sandbox(code)
                
            if success:
                logging.info(f"Execution succeeded at retry {retry_count}!")
                break
            
            logging.warning(f"Execution failed at retry {retry_count}. Error:\n{error_log}")
            retry_count += 1
            if retry_count > max_retries:
                break

            # Prompt for self-correction with traceback
            correction_prompt = f"""<|im_start|>system
You are Manas AGI. The Python code you generated previously failed or violated safety policies.
You must correct the code so that it runs successfully and prints the correct answer to standard output.

Goal: "{query}"

Failed Code:
{code}

Error Log/Traceback:
{error_log}

Rules:
1. Output ONLY valid, executable Python code. Do not wrap it in markdown code blocks like ```python. Just output the raw code.
2. The script must execute, compute the final answer, and PRINT the final result to standard output.
3. Fix the error/safety violation reported in the traceback log. Do not repeat the same bug.
4. The working directory is `/Users/zeeshanrahman/Manas/ai_agents`. All file operations must be relative to this directory. Absolute paths and parent directory traversal ('..') are strictly forbidden.
5. Do not include any conversational explanation. Only code.
<|im_end|>
<|im_start|>user
Correct the code to solve: {query}<|im_end|>
<|im_start|>assistant
"""
            code = self.llm.generate(correction_prompt, max_tokens=1000, temperature=0.1).strip()
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            code = code.strip()
            logging.info(f"Redrafted code (Retry {retry_count}):\n{code}")

        if success:
            return output, code, True
        else:
            return error_log, code, False

    def self_evaluate(self, question: str, answer: str) -> int:
        eval_prompt = f"""Rate this answer from 1 to 10 based on accuracy and completeness. Output ONLY a single integer.
Question: {question}
Answer: {answer}
Score:"""
        try:
            score_str = self.llm.generate(eval_prompt, max_tokens=10).strip()
            score = int(''.join(filter(str.isdigit, score_str)))
            return min(max(score, 1), 10)
        except:
            return 5

    def _pseudo_embed(self, text: str, dim: int = 128) -> np.ndarray:
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.randn(dim)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def _process_single_query(self, query: str) -> tuple:
        self.step_counter += 1
        query_emb = self._pseudo_embed(query)
        
        # Execute Infinite Fix Loop
        output, code, success = self.solve_via_code(query)
        
        if success:
            response_text = output.strip()
            if not response_text:
                response_text = "[Execution succeeded but printed nothing]"
            
            # Update Memory Graph
            interaction_payload = {
                "query": query,
                "code": code,
                "output": response_text,
                "status": "SUCCESS"
            }
            self.graph.add_node(
                kind=NodeKind.ANNOTATION,
                payload=interaction_payload,
                embedding=query_emb,
                step=self.step_counter,
                priority=1.0,
                tags=["execution_success", "code"]
            )
            return response_text, True
        else:
            response_text = f"Execution failed after multiple attempts.\nLast error:\n{output}"
            return response_text, False

    def chat(self, user_input: str) -> str:
        self.last_user_input_time = time.time()
        self.input_queue.put(user_input)
        
        # Extract facts from user input
        knowledge = self.understanding.extract(user_input, source="user")
        for fact in knowledge.get("facts", []):
            self.ontology.add_relation(fact.subject, fact.predicate, fact.obj)
            
        # Get AI Response via the Execution Loop
        response, success = self._process_single_query(user_input)
        
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Manas: {response}")
        
        # ONLY queue embedding for Liquid Core training ON SUCCESS!
        if success:
            emb = self._pseudo_embed(user_input + " " + response)
            self.training_queue.put((emb, True))
            
        return response

    def _logic_loop(self):
        # Continuous background learning from internet gaps
        while self.is_running:
            try:
                if self.input_queue.empty():
                    # Find gaps that have not been learned yet
                    gaps = [g for g in self.understanding.get_top_gaps(20) if g.topic not in self.learned_topics]
                    if gaps:
                        gap_topic = gaps[0].topic
                        self.learned_topics.add(gap_topic)
                        logging.info(f"Background Logic: Curious about '{gap_topic}'. Searching Wikipedia...")
                        
                        try:
                            # Actually fetch the Wikipedia page summary!
                            wiki_summary = wikipedia.summary(gap_topic, sentences=5, auto_suggest=True)
                            logging.info(f"Found Wikipedia data for '{gap_topic}'. Extracting understanding...")
                            
                            knowledge = self.understanding.extract(wiki_summary, source="wikipedia")
                            for fact in knowledge.get("facts", []):
                                self.ontology.add_relation(fact.subject, fact.predicate, fact.obj)
                            logging.info(f"Successfully learned new facts about '{gap_topic}'.")
                        except wikipedia.exceptions.DisambiguationError as e:
                            logging.warning(f"Wikipedia Disambiguation for {gap_topic}: {e.options[:3]}")
                        except wikipedia.exceptions.PageError:
                            logging.warning(f"Wikipedia Page not found for {gap_topic}.")
                        except Exception as e:
                            logging.warning(f"Wikipedia search error: {e}")
                            
                time.sleep(10)
            except Exception as e:
                logging.error(f"Logic loop error: {e}")

    def _trainer_loop(self):
        historical_embeddings = []
        while self.is_running:
            try:
                item = self.training_queue.get(timeout=2)
                emb, is_pos = item
                
                pos_emb = emb[:128]
                historical_embeddings.append(pos_emb)
                if len(historical_embeddings) > 100:
                    historical_embeddings.pop(0)
                    
                if len(historical_embeddings) > 1:
                    idx = np.random.randint(0, len(historical_embeddings)-1)
                    neg_emb = historical_embeddings[idx]
                else:
                    neg_emb = pos_emb.copy()
                    np.random.shuffle(neg_emb)
                
                self.liquid_pos_buffer.append(pos_emb)
                self.liquid_neg_buffer.append(neg_emb)
                
                if len(self.liquid_pos_buffer) >= 8:
                    px = np.array(self.liquid_pos_buffer)
                    nx = np.array(self.liquid_neg_buffer)
                    for layer in self.liquid_core:
                        layer.liquid_zulic_update(px, nx, flow_rate=0.01)
                    self.liquid_pos_buffer.clear()
                    self.liquid_neg_buffer.clear()
            except queue.Empty:
                pass
            except Exception as e:
                logging.error(f"Trainer loop error: {e}")

    def _dream_loop(self):
        # Periodic memory consolidation and pruning
        while self.is_running:
            time.sleep(600) # every 10 minutes for testing (usually 30m)
            logging.info("Starting periodic dream cycle (Memory Consolidation)...")
            try:
                # 1. Save Temporal Graph
                self.graph.save()
                
                # 2. Relax Liquid Core States (decaying state while dreaming)
                for layer in self.liquid_core:
                    layer.state *= 0.5 
                    
                logging.info(f"Dream cycle complete. Graph Nodes: {len(self.graph)}")
            except Exception as e:
                logging.error(f"Dream cycle failed: {e}")

    def shutdown(self):
        self.is_running = False
        self.self_trainer.stop()
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2)
        logging.info("Nexus 9.0 shutdown complete.")

if __name__ == "__main__":
    nexus = ManasGrandUnifiedNexus()
    print("========================================")
    print(" MANAS AGI 9.0 - ONLINE (DEEP INTEGRATION)")
    print("========================================")
    try:
        while True:
            u = input("\nYou: ")
            if u.lower() in ["exit", "quit"]:
                break
            resp = nexus.chat(u)
            print(f"\nManas: {resp}")
    except KeyboardInterrupt:
        pass
    finally:
        nexus.shutdown()
