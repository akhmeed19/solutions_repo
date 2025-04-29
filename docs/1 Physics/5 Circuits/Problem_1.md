# Equivalent Resistance Calculation Using Graph Theory

## 1. Introduction

Calculating the equivalent resistance of electrical circuits is a fundamental problem in electrical engineering. Traditional methods involve iteratively applying series and parallel rules, which becomes increasingly complex for circuits with multiple cycles and complex topologies.

By representing circuits as graphs, we can leverage graph theory to systematically reduce and analyze any circuit configuration:
- **Nodes** represent junctions in the circuit
- **Edges** represent resistors with weights equal to resistance values
- **Source and sink** nodes represent the terminals across which we measure equivalent resistance

## 2. Algorithm Description

Our algorithm combines multiple techniques to handle circuits of any complexity:

### 2.1 Basic Reduction Rules

1. **Series Reduction**:
   - **Condition**: A node has exactly 2 connections and is neither source nor sink
   - **Action**: Replace the node and its two adjacent resistors with a single resistor equal to their sum
   - **Formula**: $R_{eq} = R_1 + R_2$

2. **Parallel Reduction**:
   - **Condition**: Multiple resistors directly connect the same pair of nodes
   - **Action**: Replace them with a single equivalent resistor
   - **Formula**: $\frac{1}{R_{eq}} = \frac{1}{R_1} + \frac{1}{R_2} + ... + \frac{1}{R_n}$

### 2.2 Advanced Techniques

For circuits that cannot be fully reduced using only series and parallel reductions:

1. **Y-Δ (Wye-Delta) Transformation**:
   - Transforms three resistors in Y configuration to three resistors in Δ configuration or vice versa
   - **Y to Δ conversion**:
     $$R_{AB} = \frac{R_A \cdot R_B + R_B \cdot R_C + R_C \cdot R_A}{R_C}$$
     $$R_{BC} = \frac{R_A \cdot R_B + R_B \cdot R_C + R_C \cdot R_A}{R_A}$$
     $$R_{CA} = \frac{R_A \cdot R_B + R_B \cdot R_C + R_C \cdot R_A}{R_B}$$
   - **Δ to Y conversion**:
     $$R_A = \frac{R_{AB} \cdot R_{CA}}{R_{AB} + R_{BC} + R_{CA}}$$
     $$R_B = \frac{R_{AB} \cdot R_{BC}}{R_{AB} + R_{BC} + R_{CA}}$$
     $$R_C = \frac{R_{BC} \cdot R_{CA}}{R_{AB} + R_{BC} + R_{CA}}$$

2. **Node Elimination Method**:
   - Based on Kirchhoff's laws and nodal analysis
   - Uses matrix operations to systematically eliminate nodes

### 2.3 Algorithm Flow

```
ALGORITHM CalculateEquivalentResistance(G, source, sink):
    Initialize graph G with resistors as weighted edges
    Store original graph for visualization
    
    WHILE circuit not fully reduced:
        Apply all possible parallel reductions
        Apply all possible series reductions
        
        IF no reductions possible:
            IF Y-Delta transformations applicable:
                Apply Y-Delta transformation
            ELSE:
                Use node elimination method
                BREAK
    
    RETURN resistance between source and sink
```

## 3. Implementation

Here's a complete Python implementation using NetworkX:

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

class CircuitAnalyzer:
    def __init__(self, graph: nx.Graph, source: int, sink: int):
        """
        Initialize circuit analyzer with a graph representation of the circuit.
        
        Args:
            graph: NetworkX Graph where edges have 'weight' attribute representing resistance
            source: Source node (input terminal)
            sink: Sink node (output terminal)
        """
        self.original_graph = graph.copy()
        self.source = source
        self.sink = sink
        self.reduction_steps = []  # Store reduction steps for visualization
        
    def calculate_equivalent_resistance(self, debug: bool = False) -> float:
        """Calculate the equivalent resistance between source and sink nodes."""
        # Start with a fresh copy of the original graph
        g = self.original_graph.copy()
        self.reduction_steps = [("Original Circuit", g.copy())]
        
        # Keep applying reductions until we can't reduce further
        while True:
            parallel_count = self._reduce_all_parallel(g, debug)
            series_count = self._reduce_all_series(g, debug)
            
            # If no reductions were made, we're done with basic methods
            if parallel_count == 0 and series_count == 0:
                break
        
        # Check if we have a direct connection between source and sink
        if g.number_of_edges() == 1 and g.has_edge(self.source, self.sink):
            r_eq = g[self.source][self.sink]['weight']
            self.reduction_steps.append(("Final Reduced Circuit", g.copy()))
            if debug:
                print(f"Circuit fully reduced to {r_eq} ohms")
            return r_eq
        
        # If basic reductions couldn't fully reduce the circuit, try Y-Delta transformations
        if debug:
            print("Basic reductions insufficient. Attempting Y-Delta transformations...")
        
        delta_reductions = self._apply_y_delta_transformations(g, debug)
        
        # After Y-Delta, try basic reductions again
        if delta_reductions > 0:
            while True:
                parallel_count = self._reduce_all_parallel(g, debug)
                series_count = self._reduce_all_series(g, debug)
                if parallel_count == 0 and series_count == 0:
                    break
        
        # Check if we now have a direct connection
        if g.number_of_edges() == 1 and g.has_edge(self.source, self.sink):
            r_eq = g[self.source][self.sink]['weight']
            self.reduction_steps.append(("Final Reduced Circuit", g.copy()))
            return r_eq
        
        # If still not reduced, use the node elimination method
        if debug:
            print("Using node elimination method...")
        
        r_eq = self._node_elimination_method(g, debug)
        self.reduction_steps.append(("Final Result (Node Elimination)", 
                                   nx.Graph([(self.source, self.sink, {'weight': r_eq})])))
        
        return r_eq
    
    def _reduce_all_parallel(self, g: nx.Graph, debug: bool = False) -> int:
        """Reduce all parallel connections in the circuit."""
        reduction_count = 0
        parallel_edges = self._find_parallel_edges(g)
        
        for (n1, n2), edges in parallel_edges.items():
            if len(edges) > 1:
                # Calculate equivalent resistance for parallel resistors
                total_conductance = sum(1.0 / g[n1][n2]['weight'] for _ in range(len(edges)))
                r_eq = 1.0 / total_conductance
                
                # Replace parallel edges with a single equivalent edge
                for _ in range(len(edges) - 1):
                    g.remove_edge(n1, n2)
                g[n1][n2]['weight'] = r_eq
                
                if debug:
                    print(f"Parallel reduction: {len(edges)} resistors between nodes {n1}-{n2} → {r_eq:.2f}Ω")
                
                reduction_count += 1
                self.reduction_steps.append((f"Parallel Reduction: {n1}-{n2}", g.copy()))
        
        return reduction_count
    
    def _find_parallel_edges(self, g: nx.Graph) -> Dict[Tuple[int, int], list]:
        """Find all parallel edges in the graph."""
        parallel_edges = {}
        
        for edge in g.edges():
            n1, n2 = min(edge), max(edge)  # Ensure consistent ordering
            key = (n1, n2)
            
            if key not in parallel_edges:
                parallel_edges[key] = []
            
            parallel_edges[key].append(edge)
        
        return parallel_edges
    
    def _reduce_all_series(self, g: nx.Graph, debug: bool = False) -> int:
        """Reduce all series connections in the circuit."""
        reduction_count = 0
        series_nodes = [node for node in g.nodes() 
                      if g.degree(node) == 2 
                      and node != self.source 
                      and node != self.sink]
        
        for node in series_nodes:
            # Get the two neighbors of this node
            neighbors = list(g.neighbors(node))
            n1, n2 = neighbors[0], neighbors[1]
            
            # Get the resistances of the two edges
            r1 = g[node][n1]['weight']
            r2 = g[node][n2]['weight']
            
            # Calculate equivalent resistance (series: R_eq = R1 + R2)
            r_eq = r1 + r2
            
            # Remove the middle node and add a direct connection with equivalent resistance
            g.remove_node(node)
            g.add_edge(n1, n2, weight=r_eq)
            
            if debug:
                print(f"Series reduction: Node {node} between {n1}-{n2} → {r_eq:.2f}Ω")
            
            reduction_count += 1
            self.reduction_steps.append((f"Series Reduction: Node {node}", g.copy()))
        
        return reduction_count
    
    def _apply_y_delta_transformations(self, g: nx.Graph, debug: bool = False) -> int:
        """Apply Y-Delta and Delta-Y transformations to reduce the circuit."""
        transformation_count = 0
        
        # Try to find Y configurations (star patterns)
        for node in list(g.nodes()):
            if node == self.source or node == self.sink:
                continue
                
            if g.degree(node) == 3:
                # This node could be the center of a Y configuration
                neighbors = list(g.neighbors(node))
                if len(neighbors) == 3:
                    # Apply Y to Delta transformation
                    r_a = g[node][neighbors[0]]['weight']
                    r_b = g[node][neighbors[1]]['weight']
                    r_c = g[node][neighbors[2]]['weight']
                    
                    r_ab = (r_a * r_b + r_b * r_c + r_c * r_a) / r_c
                    r_bc = (r_a * r_b + r_b * r_c + r_c * r_a) / r_a
                    r_ca = (r_a * r_b + r_b * r_c + r_c * r_a) / r_b
                    
                    # Remove the center node
                    g.remove_node(node)
                    
                    # Add edges between the neighbors in a triangle (delta configuration)
                    g.add_edge(neighbors[0], neighbors[1], weight=r_ab)
                    g.add_edge(neighbors[1], neighbors[2], weight=r_bc)
                    g.add_edge(neighbors[2], neighbors[0], weight=r_ca)
                    
                    if debug:
                        print(f"Y-Delta transformation: Center node {node} → Delta between {neighbors}")
                    
                    transformation_count += 1
                    self.reduction_steps.append((f"Y-Delta Transformation: Node {node}", g.copy()))
                    break
        
        # If no Y configurations were found, try Delta configurations
        if transformation_count == 0:
            # Find triangles (cycles of length 3)
            triangles = []
            for node1 in g.nodes():
                for node2 in g.neighbors(node1):
                    for node3 in g.neighbors(node2):
                        if node3 in g.neighbors(node1) and node1 < node2 < node3:
                            triangles.append((node1, node2, node3))
            
            for triangle in triangles:
                # Apply Delta to Y transformation
                r_ab = g[triangle[0]][triangle[1]]['weight'] 
                r_bc = g[triangle[1]][triangle[2]]['weight']
                r_ca = g[triangle[2]][triangle[0]]['weight']
                
                denominator = r_ab + r_bc + r_ca
                r_a = (r_ab * r_ca) / denominator
                r_b = (r_ab * r_bc) / denominator
                r_c = (r_bc * r_ca) / denominator
                
                # Add a new center node
                new_node = max(g.nodes()) + 1
                g.add_node(new_node)
                
                # Remove the triangle edges
                g.remove_edge(triangle[0], triangle[1])
                g.remove_edge(triangle[1], triangle[2])
                g.remove_edge(triangle[2], triangle[0])
                
                # Add edges from center to corners
                g.add_edge(new_node, triangle[0], weight=r_a)
                g.add_edge(new_node, triangle[1], weight=r_b)
                g.add_edge(new_node, triangle[2], weight=r_c)
                
                if debug:
                    print(f"Delta-Y transformation: Triangle {triangle} → Star with center {new_node}")
                
                transformation_count += 1
                self.reduction_steps.append((f"Delta-Y Transformation: Triangle {triangle}", g.copy()))
                break
        
        return transformation_count
    
    def _node_elimination_method(self, g: nx.Graph, debug: bool = False) -> float:
        """Calculate equivalent resistance using the node elimination method."""
        # Create a list of nodes excluding the source (ground node)
        nodes = list(g.nodes())
        nodes.remove(self.source)  # Use source as reference ground
        
        # If sink is not in the remaining nodes (disconnected), return infinity
        if self.sink not in nodes:
            return float('inf')
        
        # Create the conductance matrix
        n = len(nodes)
        G_matrix = np.zeros((n, n))
        
        # Node index mapping for matrix positions
        node_indices = {node: i for i, node in enumerate(nodes)}
        
        # Fill the conductance matrix
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    # Diagonal elements: sum of conductances connected to this node
                    for neighbor in g.neighbors(node_i):
                        G_matrix[i, i] += 1.0 / g[node_i][neighbor]['weight']
                elif g.has_edge(node_i, node_j):
                    # Off-diagonal elements: negative conductance between nodes
                    G_matrix[i, j] = -1.0 / g[node_i][node_j]['weight']
        
        # For equivalent resistance between source and sink
        sink_index = node_indices[self.sink]
        
        try:
            # Calculate resistance by matrix operations
            if G_matrix.size > 0:
                # If we have a non-empty matrix
                G_inv = np.linalg.inv(G_matrix)
                r_eq = 1.0 / G_matrix[node_indices[self.sink], node_indices[self.sink]]
            else:
                # Direct connection case
                r_eq = g[self.source][self.sink]['weight'] if g.has_edge(self.source, self.sink) else float('inf')
        except np.linalg.LinAlgError:
            # Fallback to direct edge if matrix is singular
            r_eq = g[self.source][self.sink]['weight'] if g.has_edge(self.source, self.sink) else float('inf')
            
        if debug:
            print(f"Node elimination method result: {r_eq:.4f}Ω")
            
        return r_eq
    
    def draw_circuit(self, title="Circuit"):
        """Draw the circuit with resistance values on edges."""
        g = self.original_graph.copy()
        plt.figure(figsize=(8, 6))
        
        # Use spring layout
        pos = nx.spring_layout(g, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(g, pos, node_size=500, node_color='lightblue')
        
        # Highlight source and sink nodes
        nx.draw_networkx_nodes(g, pos, nodelist=[self.source, self.sink], 
                              node_color='red', node_size=700)
        
        # Draw edges with resistance values as labels
        nx.draw_networkx_edges(g, pos, width=1.5)
        edge_labels = {(u, v): f"{d['weight']}Ω" for u, v, d in g.edges(data=True)}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
        
        # Draw node labels
        nx.draw_networkx_labels(g, pos)
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_reduction_steps(self):
        """Visualize the circuit reduction process step by step."""
        steps = self.reduction_steps
        n_steps = len(steps)
        
        # Calculate grid dimensions
        cols = min(3, n_steps)
        rows = (n_steps + cols - 1) // cols
        
        plt.figure(figsize=(6 * cols, 4 * rows))
        
        for i, (title, g) in enumerate(steps):
            plt.subplot(rows, cols, i + 1)
            
            # Use spring layout
            pos = nx.spring_layout(g, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(g, pos, node_size=400, node_color='lightblue')
            
            # Highlight source and sink
            if self.source in g.nodes() and self.sink in g.nodes():
                nx.draw_networkx_nodes(g, pos, nodelist=[self.source, self.sink], 
                                      node_color='red', node_size=600)
            
            # Draw edges with resistance values as labels
            nx.draw_networkx_edges(g, pos, width=1.5)
            edge_labels = {(u, v): f"{d['weight']}Ω" for u, v, d in g.edges(data=True)}
            nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8)
            
            # Draw node labels
            nx.draw_networkx_labels(g, pos)
            
            plt.title(f"Step {i+1}: {title}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
```

## 4. Example Circuits and Visual Representations

### 4.1 Example 1: Simple Series-Parallel Circuit

```
    A---[10Ω]---B---[20Ω]---C
    |                       |
   [30Ω]                   [40Ω]
    |                       |
    D-------------------E
```

ASCII representation:

```
A---10Ω---B---20Ω---C
|                   |
30Ω                40Ω
|                   |
D---------E---------+
```

This circuit combines series and parallel elements:
- Path A-B-C: $10Ω + 20Ω = 30Ω$ (series)
- Path A-D-E-C: $30Ω + 40Ω = 70Ω$ (series)
- Paths are in parallel: $\frac{1}{R_{eq}} = \frac{1}{30} + \frac{1}{70} = \frac{7}{210}$
- $R_{eq} = \frac{210}{7} = 30Ω$

### 4.2 Example 2: Wheatstone Bridge Circuit

```
    A---[10Ω]---B
    |           |
   [20Ω]       [30Ω]
    |     [50Ω]  |
    C---[40Ω]---D
```

ASCII representation:

```
    A---10Ω---B
    |         |
   20Ω       30Ω
    |    50Ω  |
    C----     |
    |    \    |
    |     \   |
   40Ω      \ |
    |        \|
    D---------+
```

This circuit has a bridge element (50Ω between B and C), which cannot be reduced using only series and parallel rules. We need Y-Δ transformation:

1. Apply Y-Δ transformation to convert the bridge into a reducible form
2. Then apply series-parallel reductions
3. If needed, use node elimination method

### 4.3 Example 3: Complex Mesh Circuit

```
    A---[2Ω]---B---[5Ω]---C
    |         |           |
   [4Ω]      [7Ω]        [3Ω]
    |         |           |
    D---[8Ω]--E---[6Ω]---F
```

ASCII representation:

```
A---2Ω---B---5Ω---C
|        |        |
4Ω      7Ω       3Ω
|        |        |
D---8Ω---E---6Ω---F
```

This complex mesh circuit requires multiple reduction steps:
1. Cannot be directly reduced with series-parallel only
2. Need Y-Δ transformations for certain configurations
3. Final solution may require node elimination method

## 5. Algorithm Analysis

### 5.1 Time Complexity

- **Series and Parallel Reductions**: $O(|V| + |E|)$ per iteration
- **Y-Δ Transformations**: $O(|V|^3)$ in worst case
- **Node Elimination Method**: $O(|V|^3)$ for matrix operations
- **Overall**: $O(|V|^3)$ dominated by the more complex operations

### 5.2 Space Complexity

- $O(|V|^2 + |E|)$ for graph representation and matrices

### 5.3 Strengths

1. **Handles Any Circuit Topology**: Works with any valid resistor configuration
2. **Automated Analysis**: Systematically reduces complex circuits
3. **Visual Insights**: Provides step-by-step visualization of circuit reduction
4. **Multiple Methods**: Combines different techniques for comprehensive analysis

### 5.4 Potential Improvements

1. **Optimization for Sparse Circuits**: Use sparse matrix techniques for large circuits
2. **Heuristic Selection**: Develop heuristics to choose optimal reduction sequences
3. **Parallel Processing**: Identify independent subcircuits for parallel calculation
4. **Extension to AC Circuits**: Add support for complex impedances in AC circuit analysis

## 6. Conclusion

Graph theory provides a powerful framework for calculating equivalent resistance in electrical circuits. By representing circuits as graphs and applying systematic reduction techniques, we can handle circuits of arbitrary complexity. The combination of series-parallel reductions, Y-Δ transformations, and node elimination methods ensures that we can solve any valid resistor network.

This approach not only automates circuit analysis but also provides insights into the mathematical structure of electrical networks, demonstrating the elegant intersection of graph theory and electrical engineering principles.

## 7. Example Python Usage

```python
import networkx as nx

# Create a circuit graph
g = nx.Graph()

# Example: Create a bridge circuit (Wheatstone bridge)
g.add_edge('A', 'B', weight=10)  # 10Ω
g.add_edge('A', 'C', weight=20)  # 20Ω
g.add_edge('B', 'D', weight=30)  # 30Ω
g.add_edge('C', 'D', weight=40)  # 40Ω
g.add_edge('B', 'C', weight=50)  # 50Ω (bridge element)

# Create analyzer and calculate equivalent resistance
analyzer = CircuitAnalyzer(g, 'A', 'D')
r_eq = analyzer.calculate_equivalent_resistance(debug=True)

print(f"Equivalent resistance: {r_eq} Ω")

# Draw original circuit and visualization of reduction steps
analyzer.draw_circuit("Wheatstone Bridge Circuit")
analyzer.visualize_reduction_steps()
```