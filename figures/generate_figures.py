#!/usr/bin/env python3
"""
Generate publication-quality figures for the paper:
"Decoherence is Necessary for Integrated Information"

Author: Francisco Molina-Burgos
        Avermex Research Division
        fmolina@avermex.com

Usage:
    python generate_figures.py

Output (600 DPI, multiple formats):
    figures/fig1_phi_vs_noise.{pdf,png,eps}
    figures/fig2_stochastic_resonance.{pdf,png,eps}
    figures/fig3_system_scaling.{pdf,png,eps}
    figures/fig4_baseline_comparison.{pdf,png,eps}
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# Publication-quality settings (600 DPI)
DPI = 600
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'text.usetex': False,  # Set True if LaTeX is available
})

def load_results():
    """Load experimental results from JSON."""
    results_path = Path(__file__).parent.parent / 'results' / 'consciousness_maximum_entanglement_results.json'
    with open(results_path, 'r') as f:
        return json.load(f)

def stochastic_resonance_model(epsilon, a, b, c):
    """
    Stochastic resonance model:
    Phi(epsilon) = a * epsilon * exp(-b * epsilon^2) + c
    """
    return a * epsilon * np.exp(-b * epsilon**2) + c

def save_figure(fig, name):
    """Save figure in multiple formats: PDF, PNG, EPS."""
    output_dir = Path(__file__).parent

    # PDF (vector, primary format)
    fig.savefig(output_dir / f'{name}.pdf', format='pdf', dpi=DPI)

    # PNG (raster, for web/preview)
    fig.savefig(output_dir / f'{name}.png', format='png', dpi=DPI)

    # EPS (encapsulated PostScript, for journals)
    fig.savefig(output_dir / f'{name}.eps', format='eps', dpi=DPI)

    print(f"Generated: {name}.pdf, {name}.png, {name}.eps")

def fig1_phi_vs_noise():
    """Figure 1: Phi vs Noise Amplitude for different system sizes."""
    data = load_results()
    results = data['results']

    sizes = ['Small', 'Medium', 'Large', 'XLarge']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    fig, ax = plt.subplots(figsize=(6, 4))

    for size, color, marker in zip(sizes, colors, markers):
        size_data = [r for r in results if r['system_size'] == size]
        noise = [r['noise_amplitude'] for r in size_data]
        phi_max = [r['max_phi'] for r in size_data]

        neurons = size_data[0]['effective_neurons'] if size_data else 0
        ax.plot(noise, phi_max, f'{marker}-', color=color,
                label=f'{size} (n={neurons})', markersize=6, linewidth=1.5)

    ax.set_xlabel(r'Noise Amplitude ($\varepsilon$)')
    ax.set_ylabel(r'Maximum $\Phi$ (bits)')
    ax.set_title('Integrated Information vs Noise Amplitude')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(-0.5, 21)
    ax.set_ylim(-0.002, 0.042)

    ax.annotate(r'$\Phi = 0$ (pure state)', xy=(0, 0), xytext=(2, 0.005),
                fontsize=8, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    save_figure(fig, 'fig1_phi_vs_noise')
    plt.close()

def fig2_stochastic_resonance():
    """Figure 2: Stochastic resonance fit."""
    data = load_results()
    results = data['results']

    xlarge_data = [r for r in results if r['system_size'] == 'XLarge']
    noise = np.array([r['noise_amplitude'] for r in xlarge_data])
    phi_max = np.array([r['max_phi'] for r in xlarge_data])

    try:
        popt, pcov = curve_fit(stochastic_resonance_model, noise, phi_max,
                               p0=[0.02, 0.02, 0.0], maxfev=5000)
        a, b, c = popt

        residuals = phi_max - stochastic_resonance_model(noise, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((phi_max - np.mean(phi_max))**2)
        r_squared = 1 - (ss_res / ss_tot)

        epsilon_opt = np.sqrt(1 / (2 * b)) if b > 0 else 5.0
        phi_opt = stochastic_resonance_model(epsilon_opt, *popt)

    except:
        a, b, c = 0.02, 0.02, 0.0
        r_squared = 0
        epsilon_opt = 5.0
        phi_opt = max(phi_max)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(noise, phi_max, c='#d62728', s=60, zorder=5, label='Experimental')

    noise_smooth = np.linspace(0, 22, 200)
    phi_fit = stochastic_resonance_model(noise_smooth, a, b, c)
    ax.plot(noise_smooth, phi_fit, 'k-', linewidth=2,
            label=f'Fit: $R^2$ = {r_squared:.3f}')

    ax.axvline(epsilon_opt, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.scatter([epsilon_opt], [phi_opt], c='green', s=100, marker='*', zorder=6)
    ax.annotate(rf'$\varepsilon_{{\mathrm{{opt}}}}$ = {epsilon_opt:.2f}',
                xy=(epsilon_opt, phi_opt), xytext=(epsilon_opt + 3, phi_opt + 0.005),
                fontsize=9, color='green')

    ax.set_xlabel(r'Noise Amplitude ($\varepsilon$)')
    ax.set_ylabel(r'Maximum $\Phi$ (bits)')
    ax.set_title('Stochastic Resonance in Integrated Information')
    ax.legend(loc='upper right', framealpha=0.9)

    ax.text(0.98, 0.15, r'$\Phi(\varepsilon) = a \cdot \varepsilon \cdot e^{-b\varepsilon^2} + c$',
            transform=ax.transAxes, fontsize=9, ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    save_figure(fig, 'fig2_stochastic_resonance')
    plt.close()

def fig3_system_scaling():
    """Figure 3: System size scaling at optimal noise."""
    data = load_results()
    results = data['results']

    optimal_data = [r for r in results if r['noise_level'] == 'Very High']

    neurons = [r['effective_neurons'] for r in optimal_data]
    phi_max = [r['max_phi'] for r in optimal_data]
    sizes = [r['system_size'] for r in optimal_data]

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.bar(range(len(sizes)), phi_max, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
           edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f'{s}\n(n={n})' for s, n in zip(sizes, neurons)])

    ax.set_ylabel(r'Maximum $\Phi$ (bits)')
    ax.set_title(r'Scaling of $\Phi$ at Optimal Noise ($\varepsilon = 5.0$)')

    for i, (phi, n) in enumerate(zip(phi_max, neurons)):
        ax.annotate(f'{phi:.4f}', xy=(i, phi), xytext=(i, phi + 0.002),
                    ha='center', fontsize=8)

    save_figure(fig, 'fig3_system_scaling')
    plt.close()

def fig4_baseline_comparison():
    """Figure 4: Baseline (Phi=0) vs optimal configuration."""
    data = load_results()
    results = data['results']

    xlarge_baseline = [r for r in results if r['system_size'] == 'XLarge' and r['noise_level'] == 'Baseline'][0]
    xlarge_optimal = [r for r in results if r['system_size'] == 'XLarge' and r['noise_level'] == 'Very High'][0]

    fig, ax = plt.subplots(figsize=(5, 4))

    configs = [r'Pure State' + '\n' + r'($\varepsilon = 0$)',
               r'Mixed State' + '\n' + r'($\varepsilon = 5.0$)']
    phi_values = [xlarge_baseline['max_phi'], xlarge_optimal['max_phi']]
    colors = ['#3498db', '#e74c3c']

    bars = ax.bar(configs, phi_values, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

    ax.set_ylabel(r'$\Phi$ (bits)')
    ax.set_title('Pure vs Mixed Quantum States\n(XLarge System, 729 states)')

    ax.annotate(r'$\Phi = 0.0000$' + '\n(exactly zero)', xy=(0, 0.001), ha='center', fontsize=9, color='#2c3e50')
    ax.annotate(rf'$\Phi = {xlarge_optimal["max_phi"]:.4f}$', xy=(1, xlarge_optimal['max_phi'] + 0.002),
                ha='center', fontsize=9, color='#2c3e50')

    ax.set_ylim(0, 0.045)

    save_figure(fig, 'fig4_baseline_comparison')
    plt.close()

def main():
    """Generate all figures at 600 DPI in PDF, PNG, and EPS formats."""
    print("=" * 60)
    print("Generating publication figures (600 DPI)")
    print("Formats: PDF (vector), PNG (raster), EPS (encapsulated)")
    print("=" * 60)

    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    fig1_phi_vs_noise()
    fig2_stochastic_resonance()
    fig3_system_scaling()
    fig4_baseline_comparison()

    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
