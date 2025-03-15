#!/usr/bin/env python3
import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

from src.experiments.base_experiment import BaseExperiment
import src.config as config
from src.experiments.utils.logger import save_execution_log

from pyeda.inter import exprvars, truthtable  # type: ignore
from pyeda.boolalg.minimization import espresso_tts  # type: ignore

from src.tm.utils import generate_tm_input_pairs
from src.tm.machine import TuringMachine

# Experiment4 is about observing the DNF complexity of Boolean functions derived from Turing Machines (mode “tm”) or from purely random truth tables (mode “random”).
# In “tm” mode, it looks at how the transition probability influences the minimal DNF complexity of the visited-configurations function.
# In “random” mode, it simply measures the minimal DNF of random functions of the same bit-size, to see if TMs produce simpler or more complex DNFs than random.

class Experiment4(BaseExperiment):
    def __init__(self, mode, tape_length=5, num_states=4, total_bits=10, num_random_functions=2000):
        super().__init__("experiment4")
        self.mode = mode
        self.tape_length = tape_length
        self.num_states = num_states
        self.total_bits = total_bits
        self.num_random_functions = num_random_functions
    
    def measure_dnf_complexity(self, expr_obj):
        """
        Mide la complejidad de una expresión en DNF mínima.
        Se cuentan los términos (cláusulas) y el total de literales.
        """
        ast = expr_obj.to_ast()
        if isinstance(ast, tuple) and ast[0] == 'or':
            terms = ast[1:]
        else:
            terms = [ast]
        num_terms = len(terms)
        total_literals = 0
        for term in terms:
            if isinstance(term, tuple) and term[0] == 'and':
                total_literals += len(term) - 1
            else:
                total_literals += 1
        return num_terms, total_literals

    def save_frequency_percentages(self, terms_list, literals_list, mode_label):
        """
        Guarda en un archivo de texto la frecuencia y porcentaje de cada valor de términos y literales.
        """
        total = len(terms_list)
        unique_terms, counts_terms = np.unique(terms_list, return_counts=True)
        unique_literals, counts_literals = np.unique(literals_list, return_counts=True)
        output_file = os.path.join(self.run_dir, f"{mode_label}_frequency_percentages.txt")
        with open(output_file, "w") as f:
            f.write("Frecuencia y porcentajes de términos:\n")
            for ut, ct in zip(unique_terms, counts_terms):
                perc = ct / total * 100
                f.write(f"{ut}: {ct} ({perc:.2f}%)\n")
            f.write("\nFrecuencia y porcentajes de literales:\n")
            for ul, cl in zip(unique_literals, counts_literals):
                perc = cl / total * 100
                f.write(f"{ul}: {cl} ({perc:.2f}%)\n")
        self.log_message(f"Porcentajes de frecuencias ({mode_label}) guardados en: {output_file}")

    def plot_complexity_vs_probability(self, trans_probs, avg_terms, avg_literals, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(trans_probs, avg_terms, marker='o', label='Número de términos')
        plt.plot(trans_probs, avg_literals, marker='s', label='Total de literales')
        plt.xlabel('Probabilidad de transición')
        plt.ylabel('Complejidad')
        plt.title('Complejidad de la expresión mínima vs Probabilidad de transición')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_random_complexity_histogram(self, terms_list, literals_list, n_bits, title, save_path=None):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histograma de términos
        if terms_list:
            t_min = min(terms_list)
            t_max = max(terms_list)
        else:
            t_min, t_max = 0, 0
        bins_terms = np.arange(t_min, t_max + 2) - 0.5
        
        axs[0].hist(terms_list, bins=bins_terms, color='skyblue', edgecolor='black', align='mid')
        axs[0].set_title(f'Histograma de términos ({title})')
        axs[0].set_xlabel('Número de términos')
        axs[0].set_ylabel('Frecuencia')
        axs[0].grid(True)
        
        # Histograma de literales
        if literals_list:
            l_min = min(literals_list)
            l_max = max(literals_list)
        else:
            l_min, l_max = 0, 0
        bins_literals = np.arange(l_min, l_max + 2) - 0.5
        
        axs[1].hist(literals_list, bins=bins_literals, color='salmon', edgecolor='black', align='mid')
        axs[1].set_title(f'Histograma de literales ({title})')
        axs[1].set_xlabel('Total de literales')
        axs[1].set_ylabel('Frecuencia')
        axs[1].grid(True)
        
        fig.suptitle(f'Distribución de la complejidad en funciones mínimas ({n_bits} bits)')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def run_tm_mode(self):
        trans_probs = np.linspace(config.MIN_PROB, config.MAX_PROB, config.NUM_PROBS)
        
        # Estas listas guardarán un promedio por cada probabilidad
        avg_terms = []
        avg_literals = []
        
        global_terms = []
        global_literals = []
        
        # Recorremos cada probabilidad
        for p_idx, prob in enumerate(tqdm(trans_probs, desc="Modo TM", colour="blue")):
            # Para cada prob, guardamos los valores de todos los TMs
            terms_list = []
            literals_list = []
            
            machines = generate_tm_input_pairs(
                config.NUM_EXPERIMENTS,
                trans_prob=prob,
                tape_length=self.tape_length,
                num_states=self.num_states,
                total_bits=self.total_bits
            )
            
            # Recorremos las máquinas generadas con la probabilidad 'prob'
            for i, tm in enumerate(machines):
                tm.run()
                history_vals = tm.get_history_function()
                
                xs = exprvars('x', self.total_bits)
                bool_tuple = tuple(bool(x) for x in history_vals)
                tt = truthtable(xs, bool_tuple)
                min_tt, = espresso_tts(tt)
                
                if min_tt is not None:
                    n_terms, n_literals = self.measure_dnf_complexity(min_tt)
                else:
                    n_terms, n_literals = 0, 0
                
                terms_list.append(n_terms)
                literals_list.append(n_literals)
                
                global_terms.append(n_terms)
                global_literals.append(n_literals)
                
                # Guardamos un log de cada máquina
                log_data = {
                    "mode": "tm",
                    "transition_probability": prob,
                    "tm_parameters": {
                        "tape_length": tm.tape_length,
                        "num_states": tm.num_states,
                        "input_symbols": list(tm.input_symbols),
                        "blank_symbol": tm.blank_symbol,
                        "initial_head_position": tm.initial_head_position,
                        "accepting_states": tm.accepting_states
                    },
                    "input": tm.binary_input,
                    "execution": {
                        "num_steps": len(tm.config_history) - 1,
                        "result": "accepted" if tm.is_accepting() else "rejected"
                    },
                    "full_history_function": history_vals,
                    "minimized_expression": str(min_tt) if min_tt else None,
                    "complexity": {"num_terms": n_terms, "total_literals": n_literals}
                }
                filename = f"prob_{p_idx+1}_machine_{i+1}_dnf.json"
                save_execution_log(log_data, filename=filename, directory=self.run_dir)
            
            # Tras procesar todas las máquinas de esta prob, calculamos el promedio
            avg_terms.append(sum(terms_list) / len(terms_list))
            avg_literals.append(sum(literals_list) / len(literals_list))
        
        # 'avg_terms' y 'avg_literals' tienen tantos elementos como 'trans_probs'
        plot_path = os.path.join(self.run_dir, "complexity_vs_transition_probability.png")
        self.plot_complexity_vs_probability(trans_probs, avg_terms, avg_literals, save_path=plot_path)
        self.log_message(f"Modo TM completado. Gráfico guardado en: {plot_path}")

        # Histograma global (de todas las máquinas, no por probabilidad)
        hist_path = os.path.join(self.run_dir, "tm_complexity_histogram.png")
        self.plot_random_complexity_histogram(
            global_terms,
            global_literals,
            n_bits=self.tape_length + math.ceil(math.log2(self.num_states)) + math.ceil(math.log2(self.tape_length + 2)),
            title="Máquinas de Turing",
            save_path=hist_path
        )
        self.log_message(f"Histograma de complejidad TM guardado en: {hist_path}")

        # Guardar frecuencias globales
        self.save_frequency_percentages(global_terms, global_literals, mode_label="tm")

    def run_random_mode(self):
        global_terms, global_literals = [], []
        for i in tqdm(range(self.num_random_functions), desc="Random mode", colour="blue"):
            table_size = 2 ** self.total_bits
            values = [random.choice([False, True]) for _ in range(table_size)]
            xs = exprvars('x', self.total_bits)
            tt = truthtable(xs, tuple(values))
            min_tt, = espresso_tts(tt)
            if min_tt is not None:
                n_terms, n_literals = self.measure_dnf_complexity(min_tt)
            else:
                n_terms, n_literals = 0, 0

            global_terms.append(n_terms)
            global_literals.append(n_literals)

            log_data = {
                "mode": "random",
                "n_bits": self.total_bits,
                "truth_table": values,
                "minimized_expression": str(min_tt) if min_tt else None,
                "complexity": {"num_terms": n_terms, "total_literals": n_literals}
            }
            filename = f"random_function_{i+1}_dnf.json"
            save_execution_log(log_data, filename=filename, directory=self.run_dir)

        hist_path = os.path.join(self.run_dir, f"random_complexity_histogram_{self.total_bits}bits.png")
        self.plot_random_complexity_histogram(global_terms, global_literals, self.total_bits, title="Funciones aleatorias", save_path=hist_path)
        self.log_message(f"Modo Random completado. Histograma guardado en: {hist_path}")

        self.save_frequency_percentages(global_terms, global_literals, mode_label="random")

    def run_experiment(self):
        if config.EXPERIMENT4_MODE == "tm":
            # Aquí podrías parametrizar tape_length y num_states
            self.log_message("Ejecutando modo TM.")
            self.run_tm_mode(tape_length=5, num_states=4)
        elif config.EXPERIMENT4_MODE == "random":
            self.log_message(f"Ejecutando modo Random para {self.total_bits} bits.")
            self.run_random_mode(n_bits=self.total_bits, 
                                num_functions=self.num_random_functions)
        else:
            self.log_message("Modo no reconocido en config.EXPERIMENT4_MODE.")
        self.log_message("Experimento 4 completado.")

def run_experiment():
    exp = Experiment4(mode="tm", tape_length=5, num_states=4, total_bits=10, num_random_functions=2000)
    exp.run_experiment()

if __name__ == "__main__":
    run_experiment()
