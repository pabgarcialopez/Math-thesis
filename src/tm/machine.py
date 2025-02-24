# tm/machine.py

import math
from src.tm.validators import validate_params, validate_transition_params, validate_binary_input

class TuringMachine:
    def __init__(
        self,
        tape_length=5,
        num_states=4,
        input_symbols={'0', '1'},
        blank_symbol='_',
        initial_head_position=1,
        transition_function=None,
        accepting_states=None,
        binary_input=None,
        trans_prob=0.5,
        debug=False
    ):
        """
        Inicializa la Máquina de Turing.
        
        - La cinta tiene `tape_length + 2` celdas:
          índice 0: pared izquierda
          índices 1..tape_length: celdas variables
          índice tape_length+1: pared derecha
        
        - El cabezal (head) puede estar en [0..tape_length+1], aunque por defecto lo
          restringimos a la región variable [1..tape_length].
        
        - El número de estados es `num_states`. El estado inicial es 0.
        
        - Si se pasa `binary_input`, debe ser de longitud `tape_length` (bits).
          Si no se pasa, la parte variable se llena con el símbolo en blanco.
        
        - Si no se pasan `accepting_states` ni `transition_function`, se generan aleatoriamente
          con la probabilidad `trans_prob`.
        
        - Se registran todas las configuraciones (cinta+head+estado) en `config_history`.
        """
        if blank_symbol in input_symbols:
            raise ValueError("Blank symbol cannot be an input symbol.")
        
        self.num_states = num_states
        self.tape_length = tape_length
        self.input_symbols = input_symbols
        self.blank_symbol = blank_symbol
        self.binary_input = binary_input
        
        # Validar la entrada binaria si existe
        validate_binary_input(binary_input, tape_length)
        
        if binary_input is not None:
            variable_cells = list(binary_input)
        else:
            variable_cells = [blank_symbol] * tape_length
        
        # Construir la cinta: pared izquierda + celdas variables + pared derecha
        self.tape = ['|'] + variable_cells + ['|']
        
        self.head_position = initial_head_position
        
        self.current_state = 0  # Estado inicial
        
        # Generar estados aceptantes si no se pasaron
        if accepting_states is None:
            from src.tm.generators import generate_random_accepting_states
            accepting_states = generate_random_accepting_states(num_states=self.num_states)
        self.accepting_states = accepting_states
        
        # Generar función de transición si no se pasó
        if transition_function is None:
            from src.tm.generators import generate_random_transitions
            transition_function = generate_random_transitions(self, trans_prob=trans_prob)
        self.transition_function = transition_function
        
        # Historial de configuraciones
        self.config_history = []
        
        # Bits necesarios para representar la configuración
        # - tape_length bits para la parte variable de la cinta
        # - head_bits para la posición del cabezal
        # - state_bits para el estado actual
        self.head_position_bits = math.ceil(math.log2(tape_length + 2)) if (tape_length + 2) > 1 else 1
        self.state_bits = math.ceil(math.log2(num_states)) if num_states > 1 else 1
        self.total_config_bits = tape_length + self.head_position_bits + self.state_bits
        
        # Debug
        self.debug = debug

    def log(self, message):
        """Muestra un mensaje si el modo debug está activo."""
        if self.debug:
            print(message)
    
    @validate_params(validate_transition_params)
    def add_transition(self, state, symbol, next_state, write_symbol, direction):
        """
        Añade una transición a la función de transición de la TM.
        """
        self.transition_function[(state, symbol)] = (next_state, write_symbol, direction)
    
    def print_transitions(self):
        """
        Imprime la función de transición en un formato legible.
        """
        print("\nTransition Function:")
        if not self.transition_function:
            print("  No transitions defined.")
        else:
            for (state, symbol), (next_state, write_symbol, direction) in sorted(self.transition_function.items()):
                print(f"  (state {state}, symbol '{symbol}') -> (state {next_state}, write '{write_symbol}', move {direction})")
    
    def read_current_symbol(self):
        """Lee el símbolo bajo el cabezal."""
        return self.tape[self.head_position]
    
    def get_transition(self, current_symbol):
        """Devuelve la transición para (current_state, current_symbol), si existe."""
        return self.transition_function.get((self.current_state, current_symbol))
    
    def write_symbol(self, write_symbol):
        """Escribe un símbolo en la posición actual del cabezal."""
        self.tape[self.head_position] = write_symbol
    
    def move_head(self, direction):
        """
        Mueve el cabezal a la izquierda o a la derecha,
        sin salir de la región [1..tape_length].
        """
        if direction == 'R' and self.head_position < self.tape_length:
            self.head_position += 1
        elif direction == 'L' and self.head_position > 1:
            self.head_position -= 1
    
    def is_accepting(self):
        """Devuelve True si el estado actual está en los estados aceptantes."""
        return self.current_state in self.accepting_states
    
    def get_configuration(self):
        """
        Devuelve la configuración actual como cadena binaria de longitud total_config_bits:
          - tape_length bits para la parte variable de la cinta
          - head_position_bits bits para la posición del cabezal
          - state_bits bits para el estado
        """
        # Parte de la cinta (ignorando paredes)
        tape_bits = ''.join(self.tape[1:1 + self.tape_length])
        # Cabezal en binario
        head_bits = format(self.head_position, f'0{self.head_position_bits}b')
        # Estado en binario
        state_bits = format(self.current_state, f'0{self.state_bits}b')
        return tape_bits + head_bits + state_bits
    
    def step(self):
        """
        Ejecuta un paso (transición) de la MT.
        Si no hay transición definida, la máquina se detiene (accepted o rejected).
        """
        current_symbol = self.read_current_symbol()
        transition = self.get_transition(current_symbol)
        
        if not transition:
            return "accepted" if self.is_accepting() else "rejected"
        
        next_state, write_symbol, direction = transition
        self.write_symbol(write_symbol)
        self.move_head(direction)
        self.current_state = next_state
    
    def run(self, max_steps=1000):
        """
        Ejecuta la MT con un límite de pasos.
        Guarda cada configuración (cadena binaria) en self.config_history.
        Devuelve "accepted", "rejected" o "inconclusive" (si se alcanzan max_steps).
        """
        self.config_history = []
        self.config_history.append(self.get_configuration())
        
        step_count = 0
        while step_count < max_steps:
            result = self.step()
            self.config_history.append(self.get_configuration())
            if result is not None:
                return result
            step_count += 1
        
        return "accepted" if self.is_accepting() else "rejected"
    
    def get_history_function(self):
        """
        Devuelve una lista de tamaño 2^(total_config_bits).
        Se marca con 1 los índices (en decimal) de las configuraciones visitadas.
        """
        domain_size = 2 ** self.total_config_bits
        history_set = set(self.config_history)
        vec = [0] * domain_size
        for cfg in history_set:
            idx = int(cfg, 2)
            vec[idx] = 1
        return vec
    
    def get_projected_history_function(self):
        """
        Devuelve una lista de tamaño 2^(tape_length),
        marcando las configuraciones visitadas en la parte de la cinta (primeros tape_length bits).
        """
        domain_size = 2 ** self.tape_length
        projected_history_set = set(cfg[:self.tape_length] for cfg in self.config_history)
        vec = [0] * domain_size
        for i in range(domain_size):
            pattern = format(i, f'0{self.tape_length}b')
            if pattern in projected_history_set:
                vec[i] = 1
        return vec
    
    def print_tape(self, label="Tape:"):
        """
        Imprime la cinta y una marca '^' bajo la posición del cabezal.
        """
        tape_visual = ''.join(self.tape)
        line = f"{label} {tape_visual}"
        print(line)
        offset = len(label) + 1 + self.head_position
        print(" " * offset + "^")
    
    def print_history(self):
        """Imprime el historial de configuraciones (cada una en binario)."""
        print("Configuration History:")
        for config in self.config_history:
            print(config)
