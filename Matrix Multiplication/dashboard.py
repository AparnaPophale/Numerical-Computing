import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style and context for a crisp, professional look
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 16
})

# Create a multi-panel figure layout
fig = plt.figure(figsize=(15, 11), dpi=110) # Optimized dimension & DPI
fig.patch.set_facecolor('#F8F9FA')

# Define a STRICT 2-column layout for the charts to maximize width
gs = fig.add_gridspec(3, 2, height_ratios=[0.3, 1.2, 1.2], hspace=0.4, wspace=0.25)

# ----------------------------------------------------
# 1. HEADER & TITLE BLOCK
# ----------------------------------------------------
fig.suptitle('AI API Cost Simulation & Matrix Multiplication Performance Benchmark\n'
             'Institutional Infrastructure Analysis — 2,000 Student Synthetic Population',
             fontweight='bold', color='#1E293B', y=0.96)

# ----------------------------------------------------
# 2. TOP ROW: KPI CARDS (4 evenly spaced across the width)
# ----------------------------------------------------
kpi_data = [
    {"title": "TOTAL SIMULATED USERS", "val": "2,000", "color": "#475569"},
    {"title": "CHATGPT GPT-4o COST", "val": "$5,486.95", "color": "#0EA5E9"},
    {"title": "GEMINI 1.5 PRO COST", "val": "$2,743.47", "color": "#10B981"},
    {"title": "CLAUDE 3.5 SONNET COST", "val": "$7,701.50", "color": "#F59E0B"}
]

positions = [[0.06, 0.82, 0.19, 0.07], [0.29, 0.82, 0.19, 0.07], [0.52, 0.82, 0.19, 0.07], [0.75, 0.82, 0.19, 0.07]]
for i, kpi in enumerate(kpi_data):
    ax_kpi = fig.add_axes(positions[i], facecolor='#FFFFFF')
    for spine in ax_kpi.spines.values(): 
        spine.set_color('#E2E8F0')
    ax_kpi.get_xaxis().set_visible(False)
    ax_kpi.get_yaxis().set_visible(False)
    ax_kpi.text(0.5, 0.7, kpi["title"], fontsize=8, color='#64748B', weight='bold', ha='center')
    ax_kpi.text(0.5, 0.2, kpi["val"], fontsize=16, color=kpi["color"], weight='bold', ha='center')

# ----------------------------------------------------
# 3. MIDDLE ROW: BUSINESS INSIGHTS CHARTS
# ----------------------------------------------------
# Chart A: Simulated Expenditure Comparison
ax_a = fig.add_subplot(gs[1, 0])
models = ['Gemini 1.5 Pro', 'ChatGPT GPT-4o', 'Claude 3.5 Sonnet']
costs = [2743.47, 5486.95, 7701.50]
colors_a = ['#10B981', '#0EA5E9', '#F59E0B']

bars = ax_a.barh(models, costs, color=colors_a, height=0.5, edgecolor='#E2E8F0')
ax_a.set_title('Simulated Total Institutional Expenditure ($)', fontweight='bold', color='#1E293B', pad=12)
ax_a.set_xlabel('USD ($)')
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

for bar in bars:
    width = bar.get_width()
    ax_a.text(width + 150, bar.get_y() + bar.get_height()/2, f"${width:,.2f}", 
              va='center', ha='left', fontsize=9, weight='bold', color='#334155')

# Chart B: Token Demand by Academic Role (FIXED OVERLAP & WIDTH)
ax_b = fig.add_subplot(gs[1,1])
roles = ['Faculty/Staff', 'Grad Researcher', 'Undergraduate']
mean_input = [349560.04, 365932.47, 348681.02]
mean_output = [188280.24, 171500.08, 190655.93]

x = np.arange(len(roles))
width = 0.45  # Clean width allocation

# Drawing side-by-side with hard coordinate separation
rects1 = ax_b.bar(x - width/2, mean_input, width, label='Input', color='#6366F1')
rects2 = ax_b.bar(x + width/2, mean_output, width, label='Output', color='#A5B4FC')

ax_b.set_title('Average Token Volume per User', fontweight='bold', color='#1E293B', pad=12)
ax_b.set_ylabel('Tokens')
ax_b.set_xticks(x)
ax_b.set_xticklabels(roles)
ax_b.legend(frameon=True, facecolor='#FFFFFF', edgecolor='none')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

# Numeric text placement above the individual bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax_b.annotate(f'{height:,.0f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='#334155', weight='bold')

autolabel(rects1)
autolabel(rects2)
ax_b.set_ylim(0, max(mean_input) * 1.18) # Add 18% head space so labels clear the ceiling

# ----------------------------------------------------
# 4. BOTTOM ROW: ALGORITHM BENCHMARK PERFORMANCE
# ----------------------------------------------------
ax_c = fig.add_subplot(gs[2, 0])
matrix_sizes = [10, 30, 60, 100, 150, 220]
naive_times = [0.0007, 0.0159, 0.1579, 0.7393, 2.4113, 7.7118]
tiled_times = [0.0011, 0.0213, 0.1797, 0.8243, 2.6894, 8.1557]
numpy_times = [0.0001, 0.0002, 0.0004, 0.0005, 0.0009, 0.0013]

ax_c.plot(matrix_sizes, naive_times, marker='o', linewidth=2, label='Naive (Nested Loops)', color='#EF4444')
ax_c.plot(matrix_sizes, tiled_times, marker='s', linewidth=2, label='Tiled (Cache-Aware)', color='#3B82F6')
ax_c.plot(matrix_sizes, numpy_times, marker='^', linewidth=2, label='NumPy (Optimized BLAS)', color='#10B981')
ax_c.set_title('Matrix Multiplication Execution Profile', fontweight='bold', color='#1E293B', pad=12)
ax_c.set_xlabel('Matrix Dimension ($N \\times N$)')
ax_c.set_ylabel('Execution Time (seconds)')
ax_c.set_yscale('log')
ax_c.legend(frameon=True, facecolor='#FFFFFF', edgecolor='none')
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

# Chart D: Algorithmic Efficiency Gains
ax_d = fig.add_subplot(gs[2, 1])
speedup_naive = [7.8, 74.1, 383.7, 1413.0, 2726.2, 6105.9]

ax_d.plot(matrix_sizes, speedup_naive, marker='o', ls='--', linewidth=2, color='#8B5CF6')
ax_d.fill_between(matrix_sizes, speedup_naive, color='#8B5CF6', alpha=0.1)
ax_d.set_title('Hardware Acceleration Factor (NumPy Speedup)', fontweight='bold', color='#1E293B', pad=12)
ax_d.set_xlabel('Matrix Dimension ($N \\times N$)')
ax_d.set_ylabel('Speedup Multiplier ($\\|\\times\\|$)')
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)

# Key Takeaways Footer
note_text = (
    "Key Takeaways:\n\n"
    "• Python native nested loops scale cubically O(N³).\n"
    "• At N=220, NumPy outperforms native loops by ~6,105x.\n"
    "• NumPy structures exploit hardware SIMD vectorization & BLAS layers.\n"
    "• This computational speed enables scaling to 2,000+ populations smoothly."
)
fig.text(0.06, 0.01, note_text, fontsize=10, color='#334155',
         bbox=dict(facecolor='#FFFFFF', edgecolor='#E2E8F0', boxstyle='round,pad=0.8'))

plt.savefig('output/ai_api_performance_dashboard.png', dpi=120, bbox_inches='tight')
print("Dashboard image 'ai_api_performance_dashboard.png' successfully generated.")