import matplotlib.pyplot as plt
import mpld3
import numpy as np
import regex as re

# Create a Matplotlib plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [10, 20, 30], marker='o', label="Data")
ax.set_title("My Matplotlib Plot")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.legend()

# Convert the figure to HTML
html_str = mpld3.fig_to_html(fig)

figure_id = re.search(r'mpld3.draw_figure\("([\w-]+)",', html_str)
figure_id = figure_id.group(1) if figure_id else None
print(f"Figure ID: {figure_id}")

figure_data = re.search(fr'mpld3.draw_figure\("{figure_id}", (.+?)\);', html_str).group(1)
print(figure_data)