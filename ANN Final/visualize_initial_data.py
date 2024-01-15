import pandas as pd
import matplotlib.pyplot as plt

# Loading the dataset
file_path = r'C:\Users\Demirkaya\Desktop\ANN Final\myData.xlsx'
df = pd.read_excel(file_path)

# Count data points in different radius ranges
radius_ranges = [10, 20, 30, 40]
counts = []

for i in range(len(radius_ranges)):
    if i == 0:
        counts.append(len(df[df[df.columns[3]] < radius_ranges[i]]))
    else:
        counts.append(len(df[(radius_ranges[i - 1] <= df[df.columns[3]]) & (df[df.columns[3]] < radius_ranges[i])]))

# Plotting the data points
plt.figure(figsize=(10, 6))
colors = {'Y': 'blue', 'T': 'red'}
scatter = plt.scatter(df[df.columns[1]], df[df.columns[2]], c=df[df.columns[4]].map(colors), label=df[df.columns[4]])

# Drawing circles from the origin with radii of 10, 20, 30, and 40
for radius in radius_ranges:
    circle = plt.Circle((0, 0), radius, fill=False, color='black', linestyle='dashed', linewidth=1)
    plt.gca().add_patch(circle)

# Adding labels and title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Scatter Plot with Circles from Origin')

# Adding legend without numeric labels
handles, labels = scatter.legend_elements()
legend_labels = [colors[label] for label in labels]
plt.legend(handles, labels, title='Final State', labels=legend_labels)

# Adding labels indicating color code
plt.text(-40, -35, 'Y: Blue', color='blue')
plt.text(-40, -38, 'T: Red', color='red')

# Annotating the counts for each radius range
for i in range(len(radius_ranges)):
    plt.text(30, -33 - 3*i, f'Radius < {radius_ranges[i]}: {counts[i]}', color='black', fontsize=10)

# Showing the plot
plt.show()
