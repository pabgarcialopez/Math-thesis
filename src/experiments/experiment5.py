# src/experiments/experiment5.py

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pyeda.inter import exprvars, truthtable  # type: ignore
from pyeda.boolalg.minimization import espresso_tts  # type: ignore

from src.experiments.base_experiment import BaseExperiment
from src.tm.machine import TuringMachine
from src.tm.generators import generate_random_input
from src.utils.logger import save_execution_log
import src.config as config

class Experiment5(BaseExperiment):
    """
    Experiment5: prueba configuraciones concretas que dan 12, 15 y 20 bits totales.
    Se inspira en experiment4, pero aplica:
      - 12 bits => tape_length=7, num_states=2  -> 7 + 4 + 1 = 12
      - 15 bits => tape_length=9, num_states=4  -> 9 + 4 + 2 = 15
      - 20 bits => tape_length=14, num_states=4 -> 14 + 4 + 2 = 20

    Con modo 'tm' (se generan Turing Machines y se mide la complejidad de su FND mínima)
    y modo 'random' (se generan funciones booleanas aleatorias de n bits).
    """

    def __init__(self):
        super().__init__("experiment5")
        # Definimos las configuraciones para (tape_length, num_states)
        # que resultan en 12, 15 y 20 bits, respectivamente.
        self.configs = [
            # 12 bits
            {"tape_length": 7,  "num_states": 2,  "total_bits": 12},
            # 15 bits
            {"tape_length": 9,  "num_states": 4,  "total_bits": 15},
            # 20 bits
            {"tape_length": 14, "num_states": 4,  "total_bits": 20},
        ]

    # -------------------------
    # Métodos auxiliares
    # -------------------------

    def measure_dnf_complexity(self, expr_obj):
        """
        Mide la complejidad de una expresión en DNF mínima:
          - Número de términos (cláusulas en 'or')
          - Número total de literales
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

    def plot_complexity_vs_probability(self, trans_probs, avg_terms, avg_literals, save_path=None):
        """
        Dibuja la complejidad (número de términos y literales) vs probabilidad.
        """
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
        """
        Histograma doble (número de términos y número de literales).
        """
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

    def save_frequency_percentages(self, terms_list, literals_list, mode_label,
                              tape_length=None, head_bits=None, state_bits=None):
        total = len(terms_list)
        unique_terms, counts_terms = np.unique(terms_list, return_counts=True)
        unique_literals, counts_literals = np.unique(literals_list, return_counts=True)

        output_file = os.path.join(self.run_dir, f"{mode_label}_frequency_percentages.txt")
        with open(output_file, "w") as f:
            # Si tenemos info de la configuración, la escribimos
            if tape_length is not None and head_bits is not None and state_bits is not None:
                n_bits = tape_length + head_bits + state_bits
                f.write(f"Configuration:\n")
                f.write(f"  tape_length = {tape_length}\n")
                f.write(f"  head_bits   = {head_bits}\n")
                f.write(f"  state_bits  = {state_bits}\n")
                f.write(f"  total_bits  = {n_bits}\n\n")
            
            f.write("Frecuencia y porcentajes de términos:\n")
            for ut, ct in zip(unique_terms, counts_terms):
                perc = ct / total * 100
                f.write(f"{ut}: {ct} ({perc:.2f}%)\n")
            
            f.write("\nFrecuencia y porcentajes de literales:\n")
            for ul, cl in zip(unique_literals, counts_literals):
                perc = cl / total * 100
                f.write(f"{ul}: {cl} ({perc:.2f}%)\n")

        self.log_message(f"Porcentajes de frecuencias ({mode_label}) guardados en: {output_file}")


    # -------------------------
    # Lógica "random"
    # -------------------------
    def run_random_mode_for_nbits(self, n_bits, num_functions):
        """
        Genera num_functions funciones booleanas aleatorias de n_bits,
        minimiza con PyEDA, y produce histogramas y logs.
        """
        global_terms = []
        global_literals = []
        for i in tqdm(range(num_functions), desc=f"Procesando funciones de {n_bits} bits", colour="blue"):
            table_size = 2 ** n_bits
            values = [random.choice([False, True]) for _ in range(table_size)]
            
            xs = exprvars('x', n_bits)
            tt = truthtable(xs, tuple(values))
            min_tt, = espresso_tts(tt)
            if min_tt is not None:
                n_terms, n_literals = self.measure_dnf_complexity(min_tt)
            else:
                n_terms, n_literals = 0, 0
            
            global_terms.append(n_terms)
            global_literals.append(n_literals)

            # Log
            log_data = {
                "mode": "random",
                "n_bits": n_bits,
                "truth_table": values,
                "minimized_expression": str(min_tt) if min_tt else None,
                "complexity": {"num_terms": n_terms, "total_literals": n_literals}
            }
            filename = f"random_{n_bits}bits_{i+1}.json"
            save_execution_log(log_data, filename=filename, directory=self.run_dir)

        # Histograma final
        hist_path = os.path.join(self.run_dir, f"random_{n_bits}bits_histogram.png")
        self.plot_random_complexity_histogram(
            global_terms, 
            global_literals, 
            n_bits, 
            title=f"Funciones aleatorias ({n_bits} bits)",
            save_path=hist_path
        )
        self.save_frequency_percentages(global_terms, global_literals, mode_label=f"random_{n_bits}bits")

    # -------------------------
    # Lógica "TM"
    # -------------------------
    def run_tm_mode_for_config(self, tape_length, num_states, trans_probs, num_exps):
        """
        Genera TMs con (tape_length, num_states), recorre trans_probs,
        mide complejidad y produce un gráfico "complejidad vs prob" y un histograma global.
        """
        # Calculamos bits de cursor, bits de estado, total bits:
        tmp_tm = TuringMachine(tape_length, num_states)
        head_bits = tmp_tm.head_position_bits
        state_bits = tmp_tm.state_bits
        n_bits = tmp_tm.total_config_bits
        
        avg_terms = []
        avg_literals = []
        global_terms = []
        global_literals = []

        for p_idx, prob in enumerate(tqdm(trans_probs, desc=f"TM {tape_length}x{num_states}", colour="green")):
            terms_list = []
            literals_list = []

            for i in range(num_exps):
                bin_input = generate_random_input(tape_length)
                tm = TuringMachine(
                    tape_length=tape_length,
                    num_states=num_states,
                    binary_input=bin_input,
                    trans_prob=prob
                )
                tm.run()
                xs = exprvars('x', tm.total_config_bits)
                bool_tuple = tuple(bool(x) for x in tm.get_history_function())
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

                # Guardamos log
                log_data = {
                    "mode": "tm",
                    "tape_length": tape_length,
                    "num_states": num_states,
                    "prob": prob,
                    "bits_reales": tm.total_config_bits,
                    "execution": {
                        "steps": len(tm.config_history) - 1,
                        "result": "accepted" if tm.is_accepting() else "rejected"
                    },
                    "complexity": {"num_terms": n_terms, "total_literals": n_literals}
                }
                fname = f"tm_{n_bits}bits_prob{p_idx+1}_{i+1}.json"
                save_execution_log(log_data, filename=fname, directory=self.run_dir)

            avg_terms.append(sum(terms_list) / len(terms_list))
            avg_literals.append(sum(literals_list) / len(literals_list))

        # Graficamos complejidad vs prob
        plot_path = os.path.join(self.run_dir, f"complexity_vs_prob_{n_bits}bits.png")
        self.plot_complexity_vs_probability(trans_probs, avg_terms, avg_literals, save_path=plot_path)
        self.log_message(f"Gráfico de complejidad vs prob guardado en: {plot_path}")

        # Histograma global con título que muestre la configuración
        hist_title = (f"TM config: cinta={tape_length}, cursor={head_bits}, "
                    f"estados={state_bits}")
        hist_path = os.path.join(self.run_dir, f"tm_histogram_{n_bits}bits.png")
        self.plot_random_complexity_histogram(
            global_terms,
            global_literals,
            n_bits,
            title=hist_title,
            save_path=hist_path
        )

        # Guardar frecuencias, pasando la config
        self.save_frequency_percentages(global_terms, global_literals,
                                        mode_label=f"tm_{n_bits}bits",
                                        tape_length=tape_length,
                                        head_bits=head_bits,
                                        state_bits=state_bits)


    # -------------------------
    # run_experiment principal
    # -------------------------
    def run_experiment(self):
        self.log_message(f"Guardando logs en: {self.run_dir}")
        mode = config.EXPERIMENT4_MODE  # "random" o "tm"
        self.log_message(f"Experiment5 corriendo en modo '{mode}'")

        if mode == "random":
            # Para cada config, tomamos "total_bits" y generamos funciones aleatorias
            for cfg in self.configs:
                n_bits = cfg["total_bits"]
                tape_length = cfg["tape_length"]
                num_states = cfg["num_states"]
                self.log_message(f"\n[Experiment5 - random] tape_length={tape_length}, num_states={num_states} => {n_bits} bits")
                self.run_random_mode_for_nbits(n_bits, config.EXPERIMENT4_NUM_FUNCTIONS)

        elif mode == "tm":
            # Probamos cada config con un rango de probabilidades
            trans_probs = np.linspace(config.MIN_PROB, config.MAX_PROB, config.NUM_PROBS)
            for cfg in self.configs:
                tape_length = cfg["tape_length"]
                num_states = cfg["num_states"]
                n_bits = TuringMachine(tape_length, num_states).total_config_bits
                self.log_message(f"[Experiment5 - tm] tape_length={tape_length}, num_states={num_states} => {n_bits} bits")
                self.run_tm_mode_for_config(tape_length, num_states, trans_probs, config.NUM_EXPERIMENTS)
        else:
            self.log_message("Modo no reconocido en config.EXPERIMENT4_MODE.")

        self.log_message("Experiment 5 completado.")


def run_experiment():
    exp = Experiment5()
    exp.run_experiment()


if __name__ == "__main__":
    run_experiment()
