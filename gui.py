"""
GUI launcher for BERT-Augmented LLM Text Classification.
Run with: python3 gui.py
"""

import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog


SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classify_nq.py")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BERT-Augmented LLM Classification")
        self.geometry("820x720")
        self.resizable(True, True)
        self.process = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        # --- Model frame ---
        model_frame = ttk.LabelFrame(self, text="Model", padding=10)
        model_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.model_var = tk.StringVar(value="tiny")
        ttk.Radiobutton(model_frame, text="TinyLlama (no auth needed)",
                        variable=self.model_var, value="tiny").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(model_frame, text="Llama 3.2-1B-Instruct (needs HF login)",
                        variable=self.model_var, value="llama").grid(row=0, column=1, sticky="w", padx=(20, 0))

        # --- Parameters frame ---
        param_frame = ttk.LabelFrame(self, text="Parameters", padding=10)
        param_frame.pack(fill="x", padx=10, pady=5)

        # Row 0: n examples
        ttk.Label(param_frame, text="Number of examples (n):").grid(row=0, column=0, sticky="w")
        self.n_var = tk.IntVar(value=20)
        n_spin = ttk.Spinbox(param_frame, from_=1, to=10000, textvariable=self.n_var, width=8)
        n_spin.grid(row=0, column=1, sticky="w", padx=(5, 20))

        # Row 0: threshold
        ttk.Label(param_frame, text="BERT threshold:").grid(row=0, column=2, sticky="w")
        self.threshold_var = tk.DoubleVar(value=0.55)
        thresh_spin = ttk.Spinbox(param_frame, from_=0.0, to=1.0, increment=0.05,
                                  textvariable=self.threshold_var, width=8, format="%.2f")
        thresh_spin.grid(row=0, column=3, sticky="w", padx=5)

        # Row 1: max doc words
        ttk.Label(param_frame, text="Max doc words:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.max_words_var = tk.IntVar(value=500)
        words_spin = ttk.Spinbox(param_frame, from_=0, to=100000, textvariable=self.max_words_var,
                                 width=8)
        words_spin.grid(row=1, column=1, sticky="w", padx=(5, 20), pady=(8, 0))
        ttk.Label(param_frame, text="(0 = no limit)").grid(row=1, column=2, sticky="w", pady=(8, 0))

        # Row 2: seed
        ttk.Label(param_frame, text="Random seed:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.seed_var = tk.IntVar(value=42)
        seed_spin = ttk.Spinbox(param_frame, from_=0, to=99999, textvariable=self.seed_var, width=8)
        seed_spin.grid(row=2, column=1, sticky="w", padx=(5, 20), pady=(8, 0))

        # Row 2: split
        ttk.Label(param_frame, text="Split:").grid(row=2, column=2, sticky="w", pady=(8, 0))
        self.split_var = tk.StringVar(value="validation")
        split_combo = ttk.Combobox(param_frame, textvariable=self.split_var,
                                   values=["validation", "train"], width=10, state="readonly")
        split_combo.grid(row=2, column=3, sticky="w", padx=5, pady=(8, 0))

        # --- Modes frame ---
        modes_frame = ttk.LabelFrame(self, text="Evaluation Modes", padding=10)
        modes_frame.pack(fill="x", padx=10, pady=5)

        self.mode_full_var = tk.BooleanVar(value=True)
        self.mode_bert_var = tk.BooleanVar(value=True)
        self.mode_random_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(modes_frame, text="Full (no pruning — baseline)",
                        variable=self.mode_full_var).pack(side="left", padx=(0, 20))
        ttk.Checkbutton(modes_frame, text="BERT Pruning (our method)",
                        variable=self.mode_bert_var).pack(side="left", padx=(0, 20))
        ttk.Checkbutton(modes_frame, text="Random Pruning (control)",
                        variable=self.mode_random_var).pack(side="left")

        # --- Query frame ---
        query_frame = ttk.LabelFrame(self, text="Classification Query", padding=10)
        query_frame.pack(fill="x", padx=10, pady=5)

        self.query_var = tk.StringVar(
            value="Does this passage contain a direct answer to the question?"
        )
        ttk.Entry(query_frame, textvariable=self.query_var).pack(fill="x")

        # --- Output file ---
        output_frame = ttk.LabelFrame(self, text="Output", padding=10)
        output_frame.pack(fill="x", padx=10, pady=5)

        self.output_var = tk.StringVar(value="results.json")
        ttk.Entry(output_frame, textvariable=self.output_var, width=40).pack(side="left")
        ttk.Button(output_frame, text="Browse...", command=self._browse_output).pack(side="left", padx=(10, 0))

        # --- Buttons ---
        btn_frame = ttk.Frame(self, padding=5)
        btn_frame.pack(fill="x", padx=10)

        self.run_btn = ttk.Button(btn_frame, text="Run", command=self._run)
        self.run_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 10))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(btn_frame, textvariable=self.status_var,
                  foreground="gray").pack(side="right")

        # --- Log output ---
        log_frame = ttk.LabelFrame(self, text="Log", padding=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        self.log = scrolledtext.ScrolledText(log_frame, height=14, font=("Menlo", 11),
                                             bg="#1e1e1e", fg="#d4d4d4",
                                             insertbackground="white")
        self.log.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if path:
            self.output_var.set(path)

    def _build_command(self) -> list[str]:
        cmd = [sys.executable, SCRIPT]

        if self.model_var.get() == "tiny":
            cmd.append("--tiny")

        cmd += ["--n", str(self.n_var.get())]
        cmd += ["--threshold", str(self.threshold_var.get())]
        cmd += ["--max-doc-words", str(self.max_words_var.get())]
        cmd += ["--seed", str(self.seed_var.get())]
        cmd += ["--split", self.split_var.get()]
        cmd += ["--output", self.output_var.get()]

        modes = []
        if self.mode_full_var.get():
            modes.append("full")
        if self.mode_bert_var.get():
            modes.append("bert")
        if self.mode_random_var.get():
            modes.append("random")
        if modes:
            cmd += ["--modes"] + modes

        return cmd

    def _log_write(self, text: str):
        self.log.insert("end", text)
        self.log.see("end")

    def _run(self):
        modes = []
        if self.mode_full_var.get():
            modes.append("full")
        if self.mode_bert_var.get():
            modes.append("bert")
        if self.mode_random_var.get():
            modes.append("random")
        if not modes:
            self._log_write("ERROR: Select at least one evaluation mode.\n")
            return

        cmd = self._build_command()
        self.log.delete("1.0", "end")
        self._log_write(f"$ {' '.join(cmd)}\n\n")

        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Running...")

        thread = threading.Thread(target=self._run_process, args=(cmd,), daemon=True)
        thread.start()

    def _run_process(self, cmd: list[str]):
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in self.process.stdout:
                self.after(0, self._log_write, line)
            self.process.wait()
            rc = self.process.returncode
            if rc == 0:
                self.after(0, self._on_done, "Finished")
            else:
                self.after(0, self._on_done, f"Failed (exit {rc})")
        except Exception as e:
            self.after(0, self._log_write, f"\nERROR: {e}\n")
            self.after(0, self._on_done, "Error")

    def _on_done(self, msg: str):
        self.status_var.set(msg)
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.process = None

        # Try to load and display results
        output_path = self.output_var.get()
        if msg == "Finished" and os.path.exists(output_path):
            self._log_write("\n" + "=" * 60 + "\n")
            self._log_write(" RESULTS SUMMARY\n")
            self._log_write("=" * 60 + "\n")
            try:
                with open(output_path) as f:
                    results = json.load(f)
                header = f"{'Mode':<10} {'Accuracy':>10} {'F1(yes)':>10} {'LLM calls':>12} {'Time(s)':>10}\n"
                self._log_write(header)
                self._log_write("-" * 60 + "\n")
                for r in results:
                    line = (
                        f"{r['mode']:<10} "
                        f"{r['accuracy']:>10.3f} "
                        f"{r['f1_yes']:>10.3f} "
                        f"{r['llm_calls']:>12} "
                        f"{r['wall_time']:>10.1f}\n"
                    )
                    self._log_write(line)
            except Exception:
                pass

    def _stop(self):
        if self.process:
            self.process.terminate()
            self.status_var.set("Stopped")
            self._log_write("\n--- Stopped by user ---\n")
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")


if __name__ == "__main__":
    app = App()
    app.mainloop()
