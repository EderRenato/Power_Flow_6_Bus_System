"""
Newton-Raphson Power Flow Analysis
Created on Wed Apr  9 2025

@author: Eder Renato
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time

# Suprimir warnings do pandas relacionados ao acesso por posição em Series
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# Caminhos dos arquivos
line_data_path = "./Newton_Raphson_Method/line_data_of_ieee_6_bus_system.csv"
bus_data_path = "./Newton_Raphson_Method/bus_data_of_ieee_6_bus_system.csv"

# Carregamento dos dados
line_data_df = pd.read_csv(line_data_path)
bus_data_df = pd.read_csv(bus_data_path)

print("=== Análise de Fluxo de Potência ===\n")
print("Dados Iniciais das Barras:")
print(bus_data_df)
print("\nDados Iniciais das Linhas:")
print(line_data_df)


class Bus:
    """Classe representando uma barra do sistema elétrico"""
    
    def __init__(self, data):
        # Converter Series para lista para evitar warnings do pandas
        if hasattr(data, 'values'):
            data = data.values
        
        self.bus_id = int(data[0])
        self.bus_type = str(data[1]).strip().upper()
        self.voltage_amplitude = float(data[2])
        self.voltage_angle = float(data[3])
        self.load_active = float(data[4])
        self.load_reactive = float(data[5])
        self.generation_active = float(data[6])
        self.generation_reactive = float(data[7])
        
        # Validar tipo de barra
        valid_types = {'PQ', 'PV', 'SLACK'}
        if self.bus_type not in valid_types:
            raise ValueError(f"Tipo de barra inválido: {self.bus_type}")


class Line:
    """Classe representando uma linha de transmissão"""
    
    def __init__(self, data):
        # Converter Series para lista para evitar warnings do pandas
        if hasattr(data, 'values'):
            data = data.values
            
        self.origin = int(data[0])
        self.target = int(data[1])
        impedance_str = str(data[2]).replace('i', 'j')
        self.impedance = complex(impedance_str)
        
        # Validar impedância
        if abs(self.impedance) < 1e-12:
            raise ValueError(f"Impedância muito pequena na linha {self.origin}-{self.target}")
        
        self.admitance = 1.0/self.impedance
        

def build_ybus(buses, lines):
    """Monta a matriz admitância corrigida"""
    num_buses = len(buses)
    y_bus = np.zeros((num_buses, num_buses), dtype=complex)
    
    # Adicionar admitâncias das linhas (elementos fora da diagonal)
    for line in lines:
        i = line.origin - 1  # Converter para índice baseado em 0
        j = line.target - 1
        y_bus[i, j] -= line.admitance
        y_bus[j, i] -= line.admitance
    
    # Calcular elementos da diagonal (soma das admitâncias conectadas)
    for i in range(num_buses):
        y_bus[i, i] = -sum(y_bus[i, :])  # Soma de todas as admitâncias conectadas à barra i
    
    return y_bus


def calculate_power_injections(buses, ybus):
    """Calcula as potências ativas e reativas injetadas"""
    num_buses = len(buses)
    p_calc = np.zeros(num_buses)
    q_calc = np.zeros(num_buses)
    
    for i in range(num_buses):
        V_i = buses[i].voltage_amplitude
        theta_i = np.radians(buses[i].voltage_angle)
    
        for j in range(num_buses):
            V_j = buses[j].voltage_amplitude
            theta_j = np.radians(buses[j].voltage_angle)
            
            Yij = ybus[i, j]
            Gij = Yij.real
            Bij = Yij.imag
            theta_ij = theta_i - theta_j
            
            p_calc[i] += V_i * V_j * (Gij * np.cos(theta_ij) + Bij * np.sin(theta_ij))
            q_calc[i] += V_i * V_j * (Gij * np.sin(theta_ij) - Bij * np.cos(theta_ij))
    
    return p_calc, q_calc


def calculate_power_mismatches(buses, Ybus, tol=1e-6):
    """Calcula os mismatches de potência ativa e reativa"""
    num_buses = len(buses)
    P_calc, Q_calc = calculate_power_injections(buses, Ybus)
    
    delta_P = np.zeros(num_buses)
    delta_Q = np.zeros(num_buses)
    
    pq_buses = []
    pv_buses = []
    slack_bus = None
    
    # Identificar tipos de barras
    for i, bus in enumerate(buses):
        if bus.bus_type == 'PQ':
            pq_buses.append(i)
        elif bus.bus_type == 'PV':
            pv_buses.append(i)
        elif bus.bus_type == 'SLACK':
            slack_bus = i
    
    # Calcular mismatches
    for i in range(num_buses):
        P_esp = buses[i].generation_active - buses[i].load_active
        Q_esp = buses[i].generation_reactive - buses[i].load_reactive
        
        delta_P[i] = P_esp - P_calc[i]
        delta_Q[i] = Q_esp - Q_calc[i]
    
    # Definir mismatches como zero para barras apropriadas
    if slack_bus is not None:
        delta_P[slack_bus] = 0.0
        delta_Q[slack_bus] = 0.0
    
    for i in pv_buses:
        delta_Q[i] = 0.0
    
    return delta_P, delta_Q, pq_buses, pv_buses, slack_bus


def build_jacobian_matrix(buses, Ybus, pq_buses, pv_buses, slack_bus):
    """Monta a matriz Jacobiana corrigida para o método Newton-Raphson"""
    num_buses = len(buses)
    P_calc, Q_calc = calculate_power_injections(buses, Ybus)
    
    # Barras que não são slack (para equações de P)
    non_slack_buses = [i for i in range(num_buses) if i != slack_bus]
    
    num_p = len(non_slack_buses)
    num_q = len(pq_buses)
    num_total = num_p + num_q
    
    if num_total == 0:
        return np.eye(1)
    
    jacobian_matrix = np.zeros((num_total, num_total))
    
    # Submatriz H (∂P/∂θ)
    for i, bus_i in enumerate(non_slack_buses):
        V_i = buses[bus_i].voltage_amplitude
        theta_i = np.radians(buses[bus_i].voltage_angle)
        
        for j, bus_j in enumerate(non_slack_buses):
            V_j = buses[bus_j].voltage_amplitude
            theta_j = np.radians(buses[bus_j].voltage_angle)
            
            Yij = Ybus[bus_i, bus_j]
            Gij = Yij.real
            Bij = Yij.imag
            theta_ij = theta_i - theta_j
            
            if bus_i == bus_j:
                # Elemento diagonal
                jacobian_matrix[i, j] = -Q_calc[bus_i] - V_i**2 * Bij
            else:
                # Elemento fora da diagonal
                jacobian_matrix[i, j] = V_i * V_j * (Gij * np.sin(theta_ij) - Bij * np.cos(theta_ij))
    
    # Submatriz N (∂P/∂V) - apenas para barras PQ
    for i, bus_i in enumerate(non_slack_buses):
        V_i = buses[bus_i].voltage_amplitude
        theta_i = np.radians(buses[bus_i].voltage_angle)
        
        for j, bus_j in enumerate(pq_buses):
            V_j = buses[bus_j].voltage_amplitude
            theta_j = np.radians(buses[bus_j].voltage_angle)
            
            Yij = Ybus[bus_i, bus_j]
            Gij = Yij.real
            Bij = Yij.imag
            theta_ij = theta_i - theta_j
            
            if bus_i == bus_j:
                # Elemento diagonal
                jacobian_matrix[i, num_p + j] = P_calc[bus_i]/V_i + V_i * Gij
            else:
                # Elemento fora da diagonal
                jacobian_matrix[i, num_p + j] = V_i * V_j * (Gij * np.cos(theta_ij) + Bij * np.sin(theta_ij)) / V_j
    
    # Submatriz M (∂Q/∂θ) - apenas para barras PQ
    for i, bus_i in enumerate(pq_buses):
        V_i = buses[bus_i].voltage_amplitude
        theta_i = np.radians(buses[bus_i].voltage_angle)
        
        for j, bus_j in enumerate(non_slack_buses):
            V_j = buses[bus_j].voltage_amplitude
            theta_j = np.radians(buses[bus_j].voltage_angle)
            
            Yij = Ybus[bus_i, bus_j]
            Gij = Yij.real
            Bij = Yij.imag
            theta_ij = theta_i - theta_j
            
            if bus_i == bus_j:
                # Elemento diagonal
                jacobian_matrix[num_p + i, j] = P_calc[bus_i] - V_i**2 * Gij
            else:
                # Elemento fora da diagonal
                jacobian_matrix[num_p + i, j] = -V_i * V_j * (Gij * np.cos(theta_ij) + Bij * np.sin(theta_ij))
    
    # Submatriz L (∂Q/∂V) - apenas para barras PQ
    for i, bus_i in enumerate(pq_buses):
        V_i = buses[bus_i].voltage_amplitude
        theta_i = np.radians(buses[bus_i].voltage_angle)
        
        for j, bus_j in enumerate(pq_buses):
            V_j = buses[bus_j].voltage_amplitude
            theta_j = np.radians(buses[bus_j].voltage_angle)
            
            Yij = Ybus[bus_i, bus_j]
            Gij = Yij.real
            Bij = Yij.imag
            theta_ij = theta_i - theta_j
            
            if bus_i == bus_j:
                # Elemento diagonal
                jacobian_matrix[num_p + i, num_p + j] = Q_calc[bus_i]/V_i - V_i * Bij
            else:
                # Elemento fora da diagonal
                jacobian_matrix[num_p + i, num_p + j] = V_i * V_j * (Gij * np.sin(theta_ij) - Bij * np.cos(theta_ij)) / V_j
    
    return jacobian_matrix


def newton_raphson_power_flow(buses, lines, max_iterations=100, tolerance=1e-3):
    """Solução do fluxo de carga usando o método de Newton-Raphson"""
    Ybus = build_ybus(buses, lines)
    
    print(f"Matriz Admitância (Ybus):")
    print(f'Shape: {Ybus.shape}')
    print("Parte Real:")
    print(np.round(Ybus.real, 4))
    print("Parte Imaginária:")
    print(np.round(Ybus.imag, 4))
    
    print(f"\nIniciando fluxo de potência (tolerância: {tolerance:.1e})")
    print('-'*70)
    
    converged = False
    iteration = 0
    max_mismatches = []
    
    start_time = time.time()
    while iteration < max_iterations and not converged:
        iteration += 1
        
        # Calcular desbalanços de potência
        delta_P, delta_Q, pq_buses, pv_buses, slack_bus = calculate_power_mismatches(buses, Ybus, tolerance)
        
        # Construir vetor de mismatches reduzido
        non_slack_buses = [i for i in range(len(buses)) if i != slack_bus]
        delta_P_reduced = delta_P[non_slack_buses]
        delta_Q_reduced = delta_Q[pq_buses]
        delta_PQ = np.concatenate((delta_P_reduced, delta_Q_reduced))
        
        max_mismatch = np.max(np.abs(delta_PQ)) if len(delta_PQ) > 0 else 0
        max_mismatches.append(max_mismatch)
        
        print(f"Iteração {iteration}: Máximo desbalanço = {max_mismatch:.6f}")
        
        # Verificar convergência
        if max_mismatch <= tolerance:
            converged = True
            print("Convergência alcançada!")
            break
        
        # Construir matriz Jacobiana
        jacobian_matrix = build_jacobian_matrix(buses, Ybus, pq_buses, pv_buses, slack_bus)
        
        try:
            # Resolver sistema linear
            delta_x = np.linalg.solve(jacobian_matrix, delta_PQ)
        except np.linalg.LinAlgError:
            print("Matriz Jacobiana singular, usando pseudo-inversa")
            delta_x = np.linalg.pinv(jacobian_matrix) @ delta_PQ
        
        # Atualizar variáveis
        num_p = len(non_slack_buses)
        delta_theta = delta_x[:num_p]
        delta_V = delta_x[num_p:] if len(delta_x) > num_p else np.array([])
        
        # Atualizar ângulos das tensões (barras não-slack)
        for i, bus_idx in enumerate(non_slack_buses):
            buses[bus_idx].voltage_angle += np.degrees(delta_theta[i])
        
        # Atualizar magnitudes das tensões (barras PQ)
        for i, bus_idx in enumerate(pq_buses):
            if i < len(delta_V):
                new_voltage = buses[bus_idx].voltage_amplitude + delta_V[i]
                # Limitar tensão entre 0.5 e 1.5 pu
                buses[bus_idx].voltage_amplitude = max(0.5, min(1.5, new_voltage))
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"\nTempo de processamento até a convergência: {processing_time*1000:.9f} ms.")
    return converged, iteration, max_mismatches


def calculate_slack_power(buses, Ybus):
    """Calcula as potências ativa e reativa da barra slack após convergência"""
    # Encontrar a barra slack
    slack_bus_idx = None
    for i, bus in enumerate(buses):
        if bus.bus_type == 'SLACK':
            slack_bus_idx = i
            break
    
    if slack_bus_idx is None:
        print("Aviso: Nenhuma barra slack encontrada!")
        return None, None
    
    # Calcular as potências injetadas
    P_calc, Q_calc = calculate_power_injections(buses, Ybus)
    
    # Para a barra slack, a potência gerada é:
    # P_gen = P_calc + P_load
    # Q_gen = Q_calc + Q_load
    P_gen_slack = P_calc[slack_bus_idx] + buses[slack_bus_idx].load_active
    Q_gen_slack = Q_calc[slack_bus_idx] + buses[slack_bus_idx].load_reactive
    
    # Atualizar os valores na barra slack
    buses[slack_bus_idx].generation_active = P_gen_slack
    buses[slack_bus_idx].generation_reactive = Q_gen_slack
    
    return P_gen_slack, Q_gen_slack


def print_final_results(buses, converged, iteration, Ybus=None):
    """Imprime os resultados finais do fluxo de carga"""
    print("\n" + "="*90)
    if converged:
        print(f"CONVERGÊNCIA ALCANÇADA EM {iteration} ITERAÇÕES")
        
        # Calcular potência da barra slack se convergiu
        if Ybus is not None:
            P_slack, Q_slack = calculate_slack_power(buses, Ybus)
            if P_slack is not None:
                print(f"Potência calculada para a barra SLACK: P = {P_slack:.4f} pu, Q = {Q_slack:.4f} pu")
    else:
        print(f"NÃO CONVERGIU APÓS {iteration} ITERAÇÕES")
    print("="*90)
    
    print(f"{'Barra':<6} {'Tipo':<6} {'V (pu)':<10} {'θ (°)':<10} {'P_gen':<10} {'Q_gen':<10} {'P_load':<10} {'Q_load':<10}")
    print("-"*90)
    
    for bus in buses:
        # Destacar a barra slack
        if bus.bus_type == 'SLACK':
            print(f"{bus.bus_id:<6} {bus.bus_type:<6} {bus.voltage_amplitude:<10.4f} {bus.voltage_angle:<10.4f} "
                  f"{bus.generation_active:<10.4f} {bus.generation_reactive:<10.4f} "
                  f"{bus.load_active:<10.4f} {bus.load_reactive:<10.4f}")
        else:
            print(f"{bus.bus_id:<6} {bus.bus_type:<6} {bus.voltage_amplitude:<10.4f} {bus.voltage_angle:<10.4f} "
                  f"{bus.generation_active:<10.4f} {bus.generation_reactive:<10.4f} "
                  f"{bus.load_active:<10.4f} {bus.load_reactive:<10.4f}")
    
    # Resumo das potências totais do sistema
    print("\n" + "="*90)
    print("RESUMO DO SISTEMA:")
    total_p_gen = sum(bus.generation_active for bus in buses)
    total_q_gen = sum(bus.generation_reactive for bus in buses)
    total_p_load = sum(bus.load_active for bus in buses)
    total_q_load = sum(bus.load_reactive for bus in buses)
    
    print(f"Geração total:     P = {total_p_gen:.4f} pu    Q = {total_q_gen:.4f} pu")
    print(f"Carga total:       P = {total_p_load:.4f} pu    Q = {total_q_load:.4f} pu")
    print(f"Perdas do sistema: P = {(total_p_gen - total_p_load):.4f} pu    Q = {(total_q_gen - total_q_load):.4f} pu")
    print("="*90)


def plot_convergence(max_mismatches):
    """Plota o gráfico de convergência"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(max_mismatches)+1), max_mismatches, marker='o', linestyle='-', color='b')
    plt.title('Convergência do Fluxo de Carga - Método Newton-Raphson')
    plt.xlabel('Iteração')
    plt.ylabel('Máximo Desbalanço (escala logarítmica)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1e-3, color='r', linestyle='--', label='Tolerância (1e-3)')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ================== CÓDIGO PRINCIPAL ==================

try:
    # Criar objetos de barras e linhas
    buses = [Bus(row) for _, row in bus_data_df.iterrows()]
    lines = [Line(row) for _, row in line_data_df.iterrows()]
    
    print(f"\nSistema carregado: {len(buses)} barras, {len(lines)} linhas")
    
    # Verificar tipos de barras
    bus_types = [bus.bus_type for bus in buses]
    print(f"Tipos de barras: {dict(zip(range(1, len(buses)+1), bus_types))}")
    
    # Resolver fluxo de potência
    converged, iterations, max_mismatches = newton_raphson_power_flow(buses, lines, tolerance=1e-4, max_iterations=1e5)
    Ybus = build_ybus(buses, lines)  # Recalcular Ybus para exibir no final
    # Imprimir resultados (incluindo cálculo de P e Q da barra slack)
    print_final_results(buses, converged, iterations, Ybus)
    
    # Plotar convergência
    try:
        plot_convergence(max_mismatches)
    except Exception as plot_error:
        print(f"\nErro ao plotar gráfico: {plot_error}")
        print("Nota: Instale matplotlib para visualizar o gráfico de convergência")
    
    if not converged:
        print("\nSugestões para resolução de problemas:")
        print("- Verificar dados de entrada")
        print("- Verificar conectividade do sistema")
        print("- Tentar condições iniciais diferentes")
        print("- Aumentar número máximo de iterações")
        print("- Verificar se há barras isoladas")

except Exception as e:
    print(f"Erro durante execução: {e}")
    import traceback
    traceback.print_exc()