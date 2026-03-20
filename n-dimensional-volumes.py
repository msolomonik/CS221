import numpy as np
import matplotlib.pyplot as plt
from math import pi, factorial




# General formula
# Volume = C(n) * r^n

#recursive rule for C_n
# C(n) = (2*pi/n) * C(n-2)

# Plan:
# define C(n) recursively, write test cases
# write sphere vol formula
# create table or other visualization with shells to show that most of volume
# of high dimensional spheres is in the surface
# for sake of learning, re write C(n) with DP

def get_sphere_volume(radius, n):
    # Calcultes volume of sphere in n dimensions
    return C(n) * radius**n

def C(n: int) -> float:
    if n == 1:
        return 2
    elif n == 2:
        return pi
    
    return 2*pi/n * C(n-2)

def get_outer_shell_volume(radius, n, shell_size=0.01):
    # shell_size is a percent of the radius
    return  get_sphere_volume(radius, n) - get_sphere_volume(radius*(1-shell_size), n)

def get_volume_on_rim(radius, n, shell_size=0.01):
    return get_outer_shell_volume(radius, n, shell_size) / get_sphere_volume(radius, n)

def run_C_tests():
    assert(C(1) == 2)
    assert(C(2) == pi)
    assert(C(3) == 4*pi/3)
    assert(C(4) == pi**2 / 2)
    assert(C(5) == 8 *pi**2 / 15)
    assert(C(6) == pi**3 / 6)
    assert(C(7) == 16*pi**3/105)

def print_volumes():
    for i in range(1, 200):
        print("Dimension: ", i, "Rim Volume: ", get_volume_on_rim(15, i))

print_volumes()

# AI Notice:
# Everything below is AI generated, everything above is handwritten

def create_rim_visualization():
    """Create visualization showing rim percentage vs dimension for different radii"""
    
    # Set up dimensions to plot
    dimensions = range(1, 31)  # 1 to 30 dimensions
    shell_size = 0.01  # Fixed 1% shell
    
    # Different radii to compare
    radii = [0.5, 1.0, 2.0, 5.0, 10.0]
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each radius as a separate line
    for radius, color in zip(radii, colors):
        rim_percentages = []
        for n in dimensions:
            rim_pct = get_volume_on_rim(radius, n, shell_size)
            rim_percentages.append(rim_pct)
        
        plt.plot(dimensions, rim_percentages, 
                color=color, linewidth=2.5, marker='o', markersize=4,
                label=f'Radius = {radius}')
    
    plt.xlabel('Dimension', fontsize=14)
    plt.ylabel('Percent of Volume in 1% Outer Shell', fontsize=14)
    plt.title('Volume Concentration on Surface: Different Radii vs Dimension', fontsize=16, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.xlim(1, 30)
    
    # Add some annotations for key insights
    plt.text(20, 0.2, 'Notice: Rim percentage is\nindependent of radius!', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.show()

def create_shell_size_comparison():
    """Create visualization comparing different shell sizes for fixed radius"""
    
    dimensions = range(1, 31)
    radius = 1.0  # Fixed radius
    
    # Different shell sizes to compare
    shell_sizes = [0.01, 0.05, 0.1, 0.2, 0.3]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    plt.figure(figsize=(12, 8))
    
    for shell_size, color in zip(shell_sizes, colors):
        rim_percentages = []
        for n in dimensions:
            rim_pct = get_volume_on_rim(radius, n, shell_size)
            rim_percentages.append(rim_pct)
        
        plt.plot(dimensions, rim_percentages, 
                color=color, linewidth=2.5, marker='s', markersize=4,
                label=f'{shell_size*100:.0f}% shell')
    
    plt.xlabel('Dimension', fontsize=14)
    plt.ylabel('Percent of Volume in Outer Shell', fontsize=14)
    plt.title('Volume Concentration: Different Shell Sizes vs Dimension', fontsize=16, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.xlim(1, 30)
    
    plt.tight_layout()
    plt.show()

def print_some_values():
    """Print some key values to verify the calculations"""
    print("\nSample calculations:")
    print("==================")
    for dim in [2, 5, 10, 20]:
        rim_1pct = get_volume_on_rim(1.0, dim, 0.01)
        total_vol = get_sphere_volume(1.0, dim)
        print(f"Dimension {dim}: Total volume = {total_vol:.6f}, 1% rim = {rim_1pct:.1%}")

def main():
    """Main function to run visualizations"""
    print("Creating rim volume visualizations...")
    
    # Create the main visualization you requested
    create_rim_visualization()
    
    # Create a complementary visualization showing different shell sizes
    create_shell_size_comparison()
    
    # Print some values for verification
    print_some_values()

if __name__ == "__main__":
    main()