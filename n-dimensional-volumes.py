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

C_cache = {}

def C_original(n: int) -> float:
    if n == 1:
        return 2
    elif n == 2:
        return pi
    
    return 2*pi/n * C(n-2)

def C(n: int) -> float:
    if n in C_cache:
        return C_cache[n]

    if n == 1:
        C_n = 2
    elif n == 2:
        C_n = pi
    else:
        C_n = 2*pi/n * C(n-2)

    C_cache[n] = C_n
    return C_n


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


### Now that the above is done, want to show the 4 circle paradox



# AI Notice:
# Everything below is AI generated, everything above is handwritten

def compare_C_performance():
    """Compare performance between cached and original C function implementations"""
    import time
    
    print("\nPerformance Comparison: C vs C_original")
    print("=" * 45)
    
    test_dimensions = [10, 20, 30, 50, 100]
    
    for n in test_dimensions:
        # Clear cache for fair comparison
        C_cache.clear()
        
        # Time original function
        start_time = time.time()
        result_original = C_original(n)
        original_time = time.time() - start_time
        
        # Time cached function (first call builds cache)
        start_time = time.time()
        result_cached = C(n)
        cached_time = time.time() - start_time
        
        speedup = original_time / cached_time if cached_time > 0 else float('inf')
        
        print(f"Dimension {n:3d}:")
        print(f"  Original: {original_time:.6f}s")
        print(f"  Cached:   {cached_time:.6f}s (speedup: {speedup:.1f}x)")
        print(f"  Results match: {abs(result_original - result_cached) < 1e-10}")
        print()

def run_performance_demo():
    """Run the performance comparison"""
    import time
    
    compare_C_performance()
    print("Cache effectiveness demo: Computing C(50) multiple times...")
    
    # Show cache building up
    C_cache.clear()
    dimensions_to_compute = [48, 49, 50]
    
    for dim in dimensions_to_compute:
        start = time.time()
        result = C(dim)
        elapsed = time.time() - start
        print(f"C({dim}) = {result:.6f} (took {elapsed:.6f}s, cache size: {len(C_cache)})")
    
    print(f"Final cache contents: {list(C_cache.keys())}")

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

def create_absolute_volume_visualization():
    """Create visualization showing absolute sphere volume vs dimension for different radii"""
    
    dimensions = range(1, 51)  # 1 to 50 dimensions to see the peak and decline
    
    # Three radii closer together to see peak behaviors without one dominating
    radii = [1.0, 1.5, 2.0]
    colors = ['blue', 'red', 'green']
    
    plt.figure(figsize=(12, 8))
    
    # Plot each radius as a separate line
    for radius, color in zip(radii, colors):
        volumes = []
        peak_dim = 0
        peak_vol = 0
        for n in dimensions:
            vol = get_sphere_volume(radius, n)
            volumes.append(vol)
            if vol > peak_vol:
                peak_vol = vol
                peak_dim = n
        
        # Add peak info to label if it actually peaks within our range
        if peak_dim > 1 and peak_dim < 50:
            label = f'Radius = {radius} (peak at dim {peak_dim})'
        else:
            label = f'Radius = {radius}'
            
        plt.plot(dimensions, volumes, 
                color=color, linewidth=2.5, marker='o', markersize=3,
                label=label)
    
    plt.xlabel('Dimension', fontsize=14)
    plt.ylabel('Absolute Volume', fontsize=14)
    plt.title('N-Dimensional Sphere Volume vs Dimension', fontsize=16, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 50)
    
    # Add annotation pointing out the peak behavior
    plt.text(35, plt.ylim()[1]*0.7, 'All three radii show\npeak-then-decline behavior!', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
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
    print_volumes()
    
    # Show performance comparison
    run_performance_demo()

    print("Creating rim volume visualizations...")
    
    # Create the main visualization you requested
    create_rim_visualization()
    
    # Create a complementary visualization showing different shell sizes
    create_shell_size_comparison()
    
    # Create absolute volume visualization
    create_absolute_volume_visualization()
    
    # Print some values for verification
    print_some_values()

if __name__ == "__main__":
    main()
    run_C_tests()